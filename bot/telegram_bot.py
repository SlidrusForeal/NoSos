import os
import threading
import tempfile
import pandas as pd
import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime
import re
from functools import lru_cache

from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram.helpers import escape_markdown

from analytics.analytics_engine import AnalyticsEngine
from security.security_manager import SecurityManager
from utils.helpers import clean_html_tags

class TelegramBot:
    def __init__(self, config, monitor, users_file='users.csv'):
        self.users_lock = threading.Lock()
        self.track_lock = threading.Lock()
        self.monitor = monitor
        self.config = config
        self.users_file = users_file
        self.admin_id = str(config['telegram']['chat_id'])
        self.bot = Bot(token=config['telegram']['token'])
        self.app = ApplicationBuilder().token(config['telegram']['token']).build()
        self.tracked_players = defaultdict(set)
        self.player_history = defaultdict(lambda: {"x": 0, "z": 0})
        self._init_users_file()
        self._register_handlers()
        self.player_report_under_maintenance = True
        self.analytics = AnalyticsEngine(monitor)

    def _init_users_file(self):
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w', encoding='utf-8') as f:
                f.write("user_id,username,approved,subscribed\n")

    def _register_handlers(self):
        handlers = [
            CommandHandler("start", self.start),
            CommandHandler("help", self.help_command),
            CommandHandler("unsubscribe", self.unsubscribe),
            CommandHandler("approve", self.approve_user),
            CommandHandler("users", self.list_users),
            CommandHandler("send", self.send_message_command),
            CommandHandler("history", self.history),
            CommandHandler("subscribe", self.subscribe),
            CommandHandler("track", self.track_player),
            CommandHandler("untrack", self.untrack_player),
            CommandHandler("anomalies", self.anomalies),
            CommandHandler("heatmap", self.heatmap),
            CommandHandler("player_report", self.player_report),
            CommandHandler("maintance", self.maintance),
            CommandHandler("broadcast", self.broadcast_message),
            CallbackQueryHandler(self.handle_callback),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message),
        ]
        for handler in handlers:
            self.app.add_handler(handler)

    async def maintance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update, "maintance"):
            return
        self.player_report_under_maintenance = not self.player_report_under_maintenance
        status = "включён" if self.player_report_under_maintenance else "выключен"
        await update.message.reply_text(f"✅ Режим обслуживания для /player_report {status}.")

    async def _check_admin(self, update: Update, command_name: str = None) -> bool:
        user = update.effective_user
        user_id = str(user.id)
        if self.monitor.security.is_admin(user_id):
            return True
        try:
            full_name = escape_markdown(user.full_name, version=2)
            username = f"@{escape_markdown(user.username, version=2)}" if user.username else "N/A"
            command = escape_markdown(command_name, version=2) if command_name else "unknown"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            admin_alert = (
                f"🚨 *Попытка несанкционированного доступа* 🚨\n"
                f"• User ID: `{user_id}`\n"
                f"• Имя: {full_name}\n"
                f"• Username: {username}\n"
                f"• Команда: `{command}`\n"
                f"• Время: {timestamp}"
            )
            admin_alert = escape_markdown(admin_alert, version=2)
            logging.info(f"Отправка уведомления админу: {admin_alert}")
            await self.bot.send_message(
                chat_id=self.admin_id,
                text=admin_alert,
                parse_mode='MarkdownV2'
            )
            logging.warning(f"Несанкционированный доступ к команде {command} от {user_id} ({user.full_name})")
            await update.message.reply_text("⛔ Нет прав для выполнения этой команды.")
            return False
        except Exception as e:
            logging.error(f"Ошибка отправки уведомления админу: {e}")
            await update.message.reply_text("⛔ У вас нет прав для выполнения этой команды.")
            return False

    async def track_player(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        if not context.args:
            await update.message.reply_text("❌ Укажите ник игрока: /track <ник>")
            return
        player_name = " ".join(context.args).strip()
        with self.users_lock:
            users = pd.read_csv(self.users_file)
            user = users[users['user_id'] == int(user_id)]
            if user.empty or not user['approved'].values[0]:
                await update.message.reply_text("⛔️ Доступ запрещен!")
                return
        with self.track_lock:
            self.tracked_players[player_name.lower()].add(user_id)
            await update.message.reply_text(
                f"🔭 Вы подписались на перемещения игрока {escape_markdown(player_name, version=2)}\n"
                f"Используйте /untrack {escape_markdown(player_name, version=2)} для отмены",
                parse_mode='MarkdownV2'
            )

    async def untrack_player(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        if not context.args:
            await update.message.reply_text("❌ Укажите ник игрока: /untrack <ник>")
            return
        player_name = " ".join(context.args).strip().lower()
        with self.track_lock:
            if user_id in self.tracked_players.get(player_name, set()):
                self.tracked_players[player_name].remove(user_id)
                await update.message.reply_text(f"✅ Вы отписались от {player_name}")
            else:
                await update.message.reply_text("ℹ️ Вы не отслеживаете этого игрока")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            user_id = user.id
            username = user.username or ""
            avatar = None
            user_profile_photos = await context.bot.get_user_profile_photos(user_id, limit=1)
            if user_profile_photos.total_count > 0:
                avatar = user_profile_photos.photos[0][-1]
            with self.users_lock:
                with open(self.users_file, 'r+', encoding='utf-8') as f:
                    users = pd.read_csv(f)
                    if str(user_id) in users['user_id'].astype(str).values:
                        await update.message.reply_text(
                            "🛠 *Вы уже зарегистрированы в системе*",
                            parse_mode='Markdown'
                        )
                        return
                    new_user = pd.DataFrame([[user_id, username, False, True]], columns=users.columns)
                    users = pd.concat([users, new_user], ignore_index=True)
                    f.seek(0)
                    users.to_csv(f, index=False)
            await update.message.reply_text(
                "✅ *Ваш запрос отправлен администратору*",
                parse_mode='Markdown'
            )
            admin_text = (
                f"👤 *Новый запрос на доступ*\n"
                f"🆔 ID: `{user_id}`\n"
                f"📛 Имя: {user.full_name}\n"
                f"🌐 Username: @{username if username else 'N/A'}"
            )
            keyboard = [
                [
                    InlineKeyboardButton("✅ Одобрить", callback_data=f"approve_{user_id}"),
                    InlineKeyboardButton("❌ Отклонить", callback_data=f"reject_{user_id}")
                ]
            ]
            if avatar:
                await context.bot.send_photo(
                    chat_id=self.admin_id,
                    photo=avatar.file_id,
                    caption=admin_text,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode='Markdown'
                )
            else:
                await context.bot.send_message(
                    chat_id=self.admin_id,
                    text=admin_text,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode='Markdown'
                )
        except Exception as e:
            logging.error(f"Ошибка в команде /start: {e}", exc_info=True)
            await update.message.reply_text(
                "⚠️ *Произошла ошибка при обработке запроса*",
                parse_mode='Markdown'
            )

    async def approve_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update, "approve"):
            return
        if not context.args or not context.args[0].isdigit():
            await update.message.reply_text("❌ Используйте: /approve <ID>")
            return
        user_id = context.args[0]
        try:
            with self.users_lock:
                users = pd.read_csv(self.users_file)
                matching_users = users.loc[users['user_id'] == int(user_id)]
                if matching_users.empty:
                    await update.message.reply_text(f"❌ Пользователь с ID {user_id} не найден.")
                    return
                users.loc[users['user_id'] == int(user_id), 'approved'] = True
                users.to_csv(self.users_file, index=False)
            await self.bot.send_message(chat_id=user_id, text="✅ Ваш аккаунт одобрен администратором!")
            await update.message.reply_text(f"✅ Пользователь {user_id} одобрен")
            await self.bot.send_message(
                chat_id=user_id,
                text="✅ Ваш аккаунт одобрен!\n\nТеперь вам доступны команды:\n/help - Справка\n/subscribe - Подписка\n/unsubscribe - Отписка\n/history - Топ игроков\n",
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logging.error(f"Error in approve_user: {e}")
            await update.message.reply_text("Произошла ошибка при одобрении пользователя.")

    async def list_users(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update, "list_users"):
            return
        users = pd.read_csv(self.users_file)
        text = "📋 *Список пользователей:*\n"
        for _, row in users.iterrows():
            status = "✅ Одобрен" if row['approved'] else "⏳ Ожидает"
            text += f"🆔 `{row['user_id']}` | 👤 {row['username'] or 'N/A'} | {status}\n"
        await update.message.reply_text(text, parse_mode='Markdown')

    async def send_message_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update, "send"):
            return
        if not context.args or len(context.args) < 2:
            await update.message.reply_text("❌ Формат: /send <ID> <сообщение>")
            return
        target_user_id = context.args[0]
        caption = " ".join(context.args[1:]).strip()
        safe_caption = escape_markdown(caption, version=2)
        photo_file_id = None
        if update.message.photo:
            photo_file_id = update.message.photo[-1].file_id
        try:
            if photo_file_id:
                await self.bot.send_photo(
                    chat_id=target_user_id,
                    photo=photo_file_id,
                    caption=safe_caption if safe_caption else None,
                    parse_mode='MarkdownV2'
                )
            else:
                await self.bot.send_message(
                    chat_id=target_user_id,
                    text=safe_caption,
                    parse_mode='MarkdownV2'
                )
            await update.message.reply_text("✅ Сообщение отправлено!")
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        try:
            data = query.data.split('_')
            action = data[0]
            user_id = data[1]
            if action == "approve":
                with self.users_lock:
                    users = pd.read_csv(self.users_file)
                    users.loc[users['user_id'] == int(user_id), 'approved'] = True
                    users.to_csv(self.users_file, index=False)
                await self.bot.send_message(chat_id=user_id, text="✅ Ваш аккаунт одобрен!")
                await self.bot.send_message(
                    chat_id=user_id,
                    text="✅ Одобрено!\n\nДоступные команды:\n/help - Помощь\n/subscribe - Уведомления\n/unsubscribe - Отмена\n/history - Активность",
                    parse_mode=ParseMode.HTML
                )
                if query.message and query.message.text:
                    await query.edit_message_text(text=f"Пользователь {user_id} одобрен")
                else:
                    await query.message.reply_text(f"Пользователь {user_id} одобрен")
            elif action == "reject":
                with self.users_lock:
                    users = pd.read_csv(self.users_file)
                    if int(user_id) in users['user_id'].values:
                        users = users[users['user_id'] != int(user_id)]
                        users.to_csv(self.users_file, index=False)
                await self.bot.send_message(chat_id=user_id, text="❌ Ваш запрос на использование бота отклонён.")
                if query.message and query.message.text:
                    await query.edit_message_text(text=f"Пользователь {user_id} отклонён")
                else:
                    await query.message.reply_text(f"Пользователь {user_id} отклонён")
        except Exception as e:
            logging.error(f"Error in handle_callback: {e}", exc_info=True)
            await query.message.reply_text("Произошла ошибка при обработке запроса.")

    async def unsubscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        with self.users_lock:
            users = pd.read_csv(self.users_file)
            user = users[users['user_id'] == int(user_id)]
            if user.empty:
                await update.message.reply_text("❌ Вы не зарегистрированы. Используйте /start.")
                return
            if not user['subscribed'].values[0]:
                await update.message.reply_text("ℹ Вы уже отписаны от уведомлений.")
                return
            users.loc[users['user_id'] == int(user_id), 'subscribed'] = False
            users.to_csv(self.users_file, index=False)
        await update.message.reply_text("🔕 Вы отписались от уведомлений!")

    async def subscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        with self.users_lock:
            users = pd.read_csv(self.users_file)
            user = users[users['user_id'] == int(user_id)]
            if user.empty:
                await update.message.reply_text("❌ Вы не зарегистрированы. Используйте /start.")
                return
            if user['subscribed'].values[0]:
                await update.message.reply_text("ℹ Вы уже подписаны на уведомления.")
                return
            users.loc[users['user_id'] == int(user_id), 'subscribed'] = True
            users.to_csv(self.users_file, index=False)
        await update.message.reply_text("🔔 Вы подписались на уведомления!")

    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        users = pd.read_csv(self.users_file)
        if user_id not in users['user_id'].astype(str).values:
            await update.message.reply_text("❌ Вы не зарегистрированы.")
            return
        history = self.monitor.get_top_players()
        response = "📜 Топ активных игроков:\n"
        for player, time_spent in history:
            response += f"{player}: {int(time_spent // 60)} минут\n"
        await update.message.reply_text(response)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        is_admin = self.monitor.security.is_admin(user_id)
        help_text = "🛠 <b>Доступные команды</b>\n\n"
        help_text += "<b>Основные:</b>\n"
        help_text += "/help - Справка\n/subscribe - Подписка\n/unsubscribe - Отписка\n"
        help_text += "/history - Топ игроков\n/track (ник) - Трекинг игрока\n/untrack (ник) - Остановить трекинг\n"
        help_text += "/player_report (ник) - Отчёт по игроку\n\n"
        if is_admin:
            help_text += "<b>Админ:</b>\n"
            help_text += "/users - Список\n/approve - Одобрить\n/send - Сообщение\n/anomalies - Проверка\n/heatmap - Зоны\n"
        await update.message.reply_text(help_text, parse_mode='HTML')

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("ℹ Используйте /help для списка команд")

    async def anomalies(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update, "anomalies"):
            return
        if len(context.args) < 2:
            await update.message.reply_text("❌ Используйте: /anomalies <скорость> <дистанция>")
            return
        try:
            speed = float(context.args[0])
            distance = float(context.args[1])
        except ValueError:
            await update.message.reply_text("❌ Проверьте, что скорость и дистанция — числа.")
            return
        player_data = {"speed": speed, "distance": distance}
        result = self.analytics.detect_anomalies(player_data)
        await update.message.reply_text(f"Результат проверки аномалий: {result}")

    async def heatmap(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update, "heatmap"):
            return
        result = self.analytics.generate_heatmap_report()
        await update.message.reply_text(result)

    async def player_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.player_report_under_maintenance:
            await update.message.reply_text("ℹ️ Команда /player_report сейчас находится на обслуживании. Попробуйте позже.")
            return
        if not context.args:
            await update.message.reply_text("❌ Используйте: /player_report <имя игрока>")
            return
        player_name = " ".join(context.args)
        try:
            from parser.player_parser import PlayerParser
            PlayerParser.clear_cache()
            raw_report = await self.analytics.generate_player_report(player_name)
            processed_report = raw_report.replace("sports_esports", "🎮").replace("emoji_events", "🏆")
            text_report = escape_markdown(processed_report, version=2)
            MAX_LENGTH = 4096
            if len(text_report) > MAX_LENGTH:
                with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(processed_report)
                    temp_file_name = temp_file.name
                with open(temp_file_name, 'rb') as file:
                    await context.bot.send_document(
                        chat_id=update.effective_chat.id,
                        document=file,
                        filename=f"Отчёт_{player_name}.txt",
                        caption=f"📁 Полный отчёт по игроку {player_name}"
                    )
                os.unlink(temp_file_name)
            else:
                await update.message.reply_text(
                    text_report,
                    parse_mode='MarkdownV2',
                    disable_web_page_preview=True
                )
        except Exception as e:
            logging.error(f"Ошибка генерации отчёта: {e}")
            await update.message.reply_text("❌ Произошла ошибка при генерации отчёта")

    async def broadcast_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update, "broadcast"):
            return
        photo_file_id = None
        if update.message.photo:
            photo_file_id = update.message.photo[-1].file_id
        caption = " ".join(context.args).strip()
        safe_caption = escape_markdown(caption, version=2)
        users = pd.read_csv(self.users_file)
        approved_users = users[users['approved'] & users['subscribed']]
        sent_count = 0
        failed_count = 0
        for user_id in approved_users['user_id']:
            try:
                if photo_file_id:
                    await self.bot.send_photo(
                        chat_id=str(user_id),
                        photo=photo_file_id,
                        caption=safe_caption if safe_caption else None,
                        parse_mode='MarkdownV2'
                    )
                else:
                    await self.bot.send_message(
                        chat_id=str(user_id),
                        text=safe_caption,
                        parse_mode='MarkdownV2'
                    )
                sent_count += 1
                await asyncio.sleep(0.3)
            except Exception as e:
                logging.error(f"Ошибка отправки пользователю {user_id}: {e}")
                failed_count += 1
        await update.message.reply_text(
            f"📢 Сообщение отправлено!\n✅ Успешно: {sent_count}\n❌ Ошибок: {failed_count}"
        )

    def run(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.app.run_polling()

    @staticmethod
    @lru_cache(maxsize=1000)
    def _normalize_name(name: str) -> str:
        import unicodedata
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '', name).lower()
