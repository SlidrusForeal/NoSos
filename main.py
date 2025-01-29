import csv
import json
import logging
import os
import pickle
import queue
import shutil
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional
import re
import matplotlib.pyplot as plt
import pandas as pd
import requests
import unicodedata
import yaml
from matplotlib.animation import FuncAnimation
import threading
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters
)
import asyncio
import io
import random
from async_lru import alru_cache
from retrying import retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log', encoding='utf-8'),  # Добавлена кодировка
        logging.StreamHandler()
    ]
)


class AlertLevel(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass(frozen=True)
class Alert:
    message: str
    level: AlertLevel
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    cooldown: float = 60  # Добавлено поле cooldown


class BaseAlertRule(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cooldown = config.get("cooldown", 20)
        self.last_triggered: Dict[str, float] = {}
        self.alert_level = config.get("alert_level", AlertLevel.WARNING)

    @abstractmethod
    def check_conditions(self, data: Dict) -> List[Alert]:
        pass

    def _should_trigger(self, identifier: str) -> bool:
        return (time.time() - self.last_triggered.get(identifier, 0)) > self.cooldown

    def _update_cooldown(self, identifier: str):
        self.last_triggered[identifier] = time.time()


class MovementAnomalyRule(BaseAlertRule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_speed = config.get("max_speed", 50)  # Блоков в секунду
        self.teleport_threshold = config.get("teleport_threshold", 100)
        self.player_positions: Dict[str, Tuple[Tuple[float, float], float]] = {}  # История позиций игроков

    def check_conditions(self, data: Dict) -> List[Alert]:
        alerts = []
        current_time = time.time()

        for player in data.get("players", []):
            try:
                player_id = player["uuid"]
                current_pos = (player["position"]["x"], player["position"]["z"])
                last_pos, last_time = self._get_player_history(player_id)

                if last_pos is None:
                    self._update_player_history(player_id, current_pos, current_time)
                    continue

                # Расчёт расстояния и времени
                distance = self._calculate_distance(last_pos, current_pos)
                time_diff = current_time - last_time

                if time_diff > 0:
                    speed = distance / time_diff
                    if speed > self.max_speed:
                        if distance > self.teleport_threshold:
                            alert = self._create_teleport_alert(player, distance)
                        else:
                            alert = self._create_speed_alert(player, speed)

                        # Уникальный ключ: player_id + источник алерта
                        alert_id = f"{player_id}_{alert.source}"
                        if self._should_trigger(alert_id):
                            alerts.append(alert)
                            self._update_cooldown(alert_id)

                self._update_player_history(player_id, current_pos, current_time)

            except Exception as e:
                logging.error(f"Ошибка обработки игрока {player.get('name')}: {str(e)}")

        return alerts

    def _get_player_history(self, player_id: str) -> Tuple[Optional[Tuple[float, float]], float]:
        """Получение последней зафиксированной позиции игрока"""
        return self.player_positions.get(player_id, (None, 0.0))

    def _update_player_history(self, player_id: str, pos: Tuple[float, float], timestamp: float):
        """Обновление истории позиций игрока"""
        self.player_positions[player_id] = (pos, timestamp)

    @staticmethod
    def _calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Расчет расстояния между двумя точками"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def _create_speed_alert(self, player: Dict, speed: float) -> Alert:
        return Alert(
            message=f"Игрок {player['name']} движется со скоростью {speed:.1f} блоков/сек",
            level=self.alert_level,
            source="movement_anomaly",
            timestamp=datetime.now(),
            metadata={
                "player": player['name'],
                "speed": speed,
                "position": player["position"]
            },
            cooldown=self.cooldown  # Установка cooldown из правила
        )

    def _create_teleport_alert(self, player: Dict, distance: float) -> Alert:
        return Alert(
            message=f"Игрок {player['name']} переместился на {distance:.1f} блоков мгновенно",
            level=AlertLevel.CRITICAL,
            source="teleport_detection",
            timestamp=datetime.now(),
            metadata={
                "player": player['name'],
                "distance": distance,
                "from": self._get_player_history(player["uuid"])[0],
                "to": player["position"]
            },
            cooldown=self.cooldown
        )


class ZoneIntrusionRule(BaseAlertRule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.excluded = config.get("excluded", False)

    @staticmethod
    @lru_cache(maxsize=1000)
    def _normalize_name(name: str) -> str:
        """Нормализует имя для callback_data"""
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
        return re.sub(r'[^a-zA-Z0-9_]', '', name).lower()

    @lru_cache(maxsize=500)
    def _in_zone(self, x: float, z: float) -> bool:
        bounds = self.config["bounds"]
        return (bounds["xmin"] <= x <= bounds["xmax"] and
                bounds["zmin"] <= z <= bounds["zmax"])

    def check_conditions(self, data: Dict) -> List[Alert]:
        if self.excluded:
            return []

        alerts = []
        zone_name = self.config["name"]
        allowed = {self._normalize_name(p) for p in self.config.get("allowed_players", [])}

        for player in data.get("players", []):
            pos = player.get("position", {})
            x, z = pos.get("x", 0), pos.get("z", 0)

            if not self._in_zone(x, z):
                continue

            norm_name = self._normalize_name(player.get("name", ""))
            if norm_name in allowed:
                continue

            alert_id = f"{zone_name}_{norm_name}"
            if self._should_trigger(alert_id):
                alerts.append(self._create_alert(player, zone_name, x, z))
                self._update_cooldown(alert_id)

        return alerts

    def _create_alert(self, player: Dict, zone_name: str, x: float, z: float) -> Alert:
        return Alert(
            message=f"Игрок {player.get('name', 'Unknown')} в зоне {zone_name}",
            level=self.alert_level,
            source="zone_intrusion",
            timestamp=datetime.now(),
            metadata={
                "player": player.get('name'),
                "zone": zone_name,
                "coordinates": (x, z)
            },
            cooldown=self.cooldown  # Установка cooldown из правила
        )


class PlayerCountRule(BaseAlertRule):
    def check_conditions(self, data: Dict) -> List[Alert]:
        current = len(data.get("players", []))
        max_players = self.config.get("max_players", 50)
        return [Alert(
            message=f"Превышен лимит игроков: {current}/{max_players}",
            level=AlertLevel.CRITICAL,
            source="player_limit",
            timestamp=datetime.now(),
            metadata={"current": current, "max": max_players}
        )] if current > max_players else []


class AlertManager:
    def __init__(self, monitor):  # <-- Добавляем параметр
        self.monitor = monitor
        self.rules: List[BaseAlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=1000)

    def get_active_alerts(self) -> List[Alert]:
        """Возвращает список активных алертов"""
        return list(self.active_alerts.values())

    def _generate_alert_id(self, alert: Alert) -> str:
        """Генерация уникального ID для алерта на основе его данных."""
        return f"{alert.source}_{alert.timestamp.timestamp()}_{alert.metadata.get('player', '')}"

    def load_rules_from_config(self, config: Dict):
        rule_registry = {
            "zone_intrusion": ZoneIntrusionRule,
            "player_limit": PlayerCountRule,
            "movement_anomaly": MovementAnomalyRule
        }

        for rule_config in config.get("zones", []):
            self.rules.append(rule_registry["zone_intrusion"](rule_config))

        if "limits" in config:
            self.rules.append(rule_registry["player_limit"](config["limits"]))

        if "movement_anomaly" in config:
            self.rules.append(rule_registry["movement_anomaly"](config["movement_anomaly"]))

    def process_data(self, data: Dict):
        new_alerts = []

        for rule in self.rules:
            try:
                new_alerts.extend(rule.check_conditions(data))
            except Exception as e:
                logging.error(f"Ошибка в правиле {rule.__class__.__name__}: {str(e)}")

        for alert in new_alerts:
            alert_id = self._generate_alert_id(alert)  # <-- Правильный вызов
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.monitor.telegram_bot.sync_send_alert(alert)

        self._clean_expired_alerts()

    def _clean_expired_alerts(self):
        current_time = time.time()
        to_remove = []
        for alert_id, alert in self.active_alerts.items():
            expire_time = alert.timestamp.timestamp() + alert.cooldown
            if current_time > expire_time:
                to_remove.append(alert_id)
        for aid in to_remove:
            del self.active_alerts[aid]

    def get_alert_history(self, limit=50) -> List[Alert]:
        return list(self.alert_history)[-limit:]


class SecurityManager:
    def __init__(self, config):
        self.config = config['security']
        self.log_file = self.config['log_file']

    def is_admin(self, user_id: str) -> bool:
        return str(user_id) in self.config['admins']

    def log_event(self, event_type: str, user_id: str, command: str):
        entry = f"{datetime.now().isoformat()} | {event_type} | User: {user_id} | Command: {command}\n"
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(entry)


class BackupSystem:
    def __init__(self):
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)

    async def daily_backup(self):
        while True:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                backup_path = f"{self.backup_dir}/activity_{timestamp}.db"
                shutil.copy2(self.monitor.config['database']['filename'], backup_path)
                logging.info(f"Backup created: {backup_path}")
            except Exception as e:
                logging.error(f"Backup error: {str(e)}")
            await asyncio.sleep(86400)  # 24 часа


class AnalyticsEngine:
    def __init__(self, monitor):
        self.monitor = monitor
        self.config = monitor.config.get("analytics", {})
        self.max_speed = self.config.get("max_speed", 50)
        self.teleport_threshold = self.config.get("teleport_threshold", 100)

    def detect_anomalies(self, player_data: dict) -> str:
        """Обнаружение аномалий в данных игрока."""
        anomalies = []

        # Проверка скорости
        if player_data.get("speed", 0) > self.max_speed:
            anomalies.append("Высокая скорость перемещения")

        # Проверка телепортаций
        if player_data.get("distance", 0) > self.teleport_threshold:
            anomalies.append("Подозрительная телепортация")

        # Проверка времени в зонах
        zone_time = self.monitor.zone_time.get(player_data["name"], {})
        for zone, time_spent in zone_time.items():
            if time_spent > self.config.get("max_zone_time", 3600):
                anomalies.append(f"Слишком долгое пребывание в зоне '{zone}'")

        return " | ".join(anomalies) if anomalies else "Норма"

    def generate_heatmap_report(self) -> str:
        """Генерация отчёта по активным зонам."""
        zone_activity = {}
        for zone, players in self.monitor.zone_time.items():
            total_time = sum(players.values())
            zone_activity[zone] = total_time

        sorted_zones = sorted(zone_activity.items(), key=lambda x: x[1], reverse=True)[:5]

        report = "🔥 Топ-5 активных зон:\n"
        for zone, time_spent in sorted_zones:
            report += f"• {zone}: {time_spent // 60} минут\n"

        return report

    def generate_player_report(self, player_name: str) -> str:
        """Отчёт по активности конкретного игрока."""
        total_time = self.monitor.player_time.get(player_name, 0)
        zone_time = self.monitor.zone_time.get(player_name, {})

        report = (
            f"📊 Отчёт по игроку *{player_name}*\n"
            f"Общее время онлайн: {total_time // 3600} ч. {total_time % 3600 // 60} мин.\n"
            "Время в зонах:\n"
        )

        for zone, time in zone_time.items():
            report += f"• {zone}: {time // 60} мин.\n"

        return report
class TelegramBot:
    def __init__(self, config, monitor, users_file='users.csv'):  # Добавлен параметр monitor
        self.monitor = monitor  # Теперь monitor передается явно
        if not config.get('telegram'):
            raise ValueError("Отсутствует секция telegram в конфиге")
        self.config = config
        self.users_file = users_file
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                f.write("user_id,username,approved\n")
        self.admin_id = str(config['telegram']['chat_id'])
        self.bot = Bot(token=config['telegram']['token'])
        self.admin_id = str(config['telegram']['chat_id'])  # Приводим к строке
        self.app = ApplicationBuilder().token(config['telegram']['token']).build()


        # Регистрация обработчиков
        self.app.add_handler(CallbackQueryHandler(self.handle_tracking_callback, pattern="^track_"))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("approve", self.approve_user))
        self.app.add_handler(CommandHandler("users", self.list_users))
        self.app.add_handler(CommandHandler("unsubscribe", self.unsubscribe))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("caramel_pain", self.caramel_pain_command))
        self.app.add_handler(CommandHandler("track", self.track_player))
        self.app.add_handler(CommandHandler("send", self.send_message_command))
        self.app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND | filters.Entity("bot_command")), self.handle_message))

    async def _check_admin(self, update: Update, command: str) -> bool:
        user_id = str(update.effective_user.id)
        is_admin = self.monitor.security.is_admin(user_id)

        if not is_admin:
            self.monitor.security.log_event("UNAUTHORIZED_ACCESS", user_id, command)
            await update.message.reply_text("⛔ Ты адекатная? А ничо тот факт что ты не администратор бота и у тебя жижа за 50 рублей купленая у ашота. \n жди докс короче")
            await self._notify_admins_about_breach(user_id, command)

        return is_admin

    async def _notify_admins_about_breach(self, user_id: str, command: str):
        message = f"🚨 Попытка несанкционированного доступа требуется докс:\nUser ID: {user_id}\nCommand: {command}"
        for admin_id in self.monitor.config['security']['admins']:
            try:
                await self.bot.send_message(chat_id=admin_id, text=message)
            except Exception as e:
                logging.error(f"Can't notify admin {admin_id}: {str(e)}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        users = pd.read_csv(self.users_file)

        if user_id in users['user_id'].values:
            await update.message.reply_text("🛠 Вы уже зарегистрированы в системе")
        else:
            new_user = pd.DataFrame([[user_id, "", False]], columns=users.columns)
            users = pd.concat([users, new_user], ignore_index=True)
            users.to_csv(self.users_file, index=False)
            await update.message.reply_text("✅ Ваш запрос отправлен администратору на одобрение")

            # Уведомление админа
            keyboard = [[InlineKeyboardButton("✅ Одобрить", callback_data=f"approve_{user_id}")]]
            await self.bot.send_message(
                chat_id=self.admin_id,
                text=f"⚠ Новый запрос на доступ:\nID: {user_id}",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        logging.warning(f"Необработанный callback: {query.data}")

        if query.data.startswith("approve_"):
            # Проверка прав администратора
            if str(query.from_user.id) != self.admin_id:
                await query.message.reply_text("⛔ Ты адекатная? А ничо тот факт что ты не администратор бота и у тебя жижа за 50 рублей купленная у ашота. \n жди докс короче")
                return

            try:
                user_id = int(query.data.split("_")[1])
                users = pd.read_csv(self.users_file)

                # Обновляем статус пользователя
                if user_id in users['user_id'].values:
                    users.loc[users['user_id'] == user_id, 'approved'] = True
                    users.to_csv(self.users_file, index=False)

                    # Уведомление администратора
                    await query.message.edit_text(f"✅ Пользователь {user_id} одобрен")

                    # Попытка уведомить пользователя
                    try:
                        await self.bot.send_message(
                            chat_id=user_id,
                            text=f"🎉 Ваш аккаунт одобрен!\n{self._get_user_commands()}",
                            parse_mode='Markdown'
                        )
                    except Exception as e:
                        logging.error(f"Ошибка уведомления пользователя: {str(e)}")
                        await query.message.reply_text("⚠ Не удалось отправить уведомление пользователю")
                else:
                    await query.message.reply_text("❌ Пользователь не найден в базе")

            except Exception as e:
                logging.error(f"Callback error: {str(e)}")
                await query.message.reply_text("⚠ Ошибка обработки запроса")

    async def send_message_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда для отправки сообщений от имени бота (только для админов)"""
        if not await self._check_admin(update, "/send"):
            return

        if not context.args or len(context.args) < 2:
            await update.message.reply_text("❌ Формат: /send <ID_пользователя> <текст сообщения>")
            return

        try:
            user_id = context.args[0]
            message = " ".join(context.args[1:])

            await self.bot.send_message(
                chat_id=user_id,
                text=f"🔔 Сообщение от администратора:\n\n{message}"
            )
            await update.message.reply_text(f"✅ Сообщение отправлено пользователю {user_id}")

            # Логирование действия
            self.monitor.security.log_event("ADMIN_MESSAGE", update.effective_user.id, f"to {user_id}: {message}")

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")
            logging.error(f"Send message error: {str(e)}")

    def _get_user_commands(self) -> str:
        return """
        📚 *Основные команды*:
        /start - Запросить доступ к боту
        /help - Список всех команд
        /unsubscribe - Отписаться от уведомлений
        /stats - Персональная статистика активности
        /history - История перемещений (пример: history K1zik)
        /track - Трекинг игрока (пример: track K1zik)
        """

    async def approve_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update, "/approve"):
            return

        try:
            user_id = context.args[0]
            users = pd.read_csv(self.users_file)
            users.loc[users['user_id'] == int(user_id), 'approved'] = True
            users.to_csv(self.users_file, index=False)

            # Отправляем пользователю список команд
            await self.bot.send_message(
                chat_id=user_id,
                text=f"🎉 Ваш аккаунт одобрен!\n{self._get_user_commands()}",
                parse_mode='Markdown'
            )
            await update.message.reply_text(f"✅ Пользователь {user_id} одобрен")

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")

    async def list_users(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update, "/users"):
            return

        users = pd.read_csv(self.users_file)
        approved_users = users[users['approved']]
        text = "📋 Список одобренных пользователей:\n" + "\n".join(
            f"ID: {row['user_id']} | Username: {row['username']}"
            for _, row in approved_users.iterrows()
        )
        await update.message.reply_text(text)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("ℹ Используйте /help для списка команд")

    async def caramel_pain_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        secret_phrases = [
            "Shin Sei Moku Roku",
            "La-Li-Lu-Le-Lo",
            "Кто такие мышериоты?"
        ]
        response = random.choice(secret_phrases)
        await update.message.reply_text(f"🔐 {response}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
        📚 *Основные команды*:
        /start - Запросить доступ к боту
        /help - Показать список команд
        /unsubscribe - Отписаться от уведомлений
        /stats - Статистика активности
        /history - История перемещений игрока
        /track - Трекинг игрока (пример: track K1zik)
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.monitor.activity_by_hour:
            await update.message.reply_text("📊 Данные статистики отсутствуют")
            return

        self.monitor.gui_queue.put(
            (self.monitor.generate_stats_plot, (update,), time.time())
        )

    async def track_player(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Трекинг конкретного игрока с интерактивным меню"""
        if not context.args:
            await update.message.reply_text("ℹ Укажите ник игрока: /track <ник>")
            return

        player_name = context.args[0].strip()
        user_id = update.effective_user.id

        # Получение данных
        tracking_data = self.monitor.get_player_tracking_data(player_name)

        if not tracking_data['current_pos']:
            await update.message.reply_text(f"❌ Игрок {player_name} не найден или не активен")
            return

        # Создание интерактивного меню
        keyboard = [
            [InlineKeyboardButton("📍 Текущая позиция", callback_data=f"track_pos_{player_name}"),
             InlineKeyboardButton("🔄 История перемещений", callback_data=f"track_history_{player_name}")],
            [InlineKeyboardButton("🔔 Подписаться на обновления", callback_data=f"track_subscribe_{player_name}")]
        ]

        message = (
            f"🎯 Трекинг игрока *{player_name}*\n"
            f"Последнее обновление: {datetime.fromtimestamp(tracking_data['last_seen']).strftime('%Y-%m-%d %H:%M')}\n"
            f"Всего записей в истории: {len(tracking_data['path_history'])}"
        )

        await update.message.reply_text(
            message,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Нормализация имени игрока для callback_data"""
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
        return re.sub(r'[^a-zA-Z0-9_]', '', name).lower()

    async def handle_tracking_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        try:
            await query.answer()
            logging.info(f"Обработка callback: {query.data}")

            if not query.data.startswith("track_"):
                return

            parts = query.data.split('_')
            if len(parts) < 3:
                raise ValueError("Некорректный формат callback_data")

            action = parts[1]
            player_name = '_'.join(parts[2:])

            # Используем метод нормализации из TelegramBot
            normalized_name = self._normalize_name(player_name)  # Теперь метод доступен
            tracking_data = self.monitor.get_player_tracking_data(normalized_name)

            if not tracking_data['current_pos']:
                await query.edit_message_text("❌ Позиция игрока недоступна")
                return

            x, z = tracking_data['current_pos']
            if x is None or z is None:
                await query.edit_message_text("❌ Координаты не найдены")
                return

            # Обработка действий
            if action == "pos":
                x, z = tracking_data['current_pos']
                text = f"📍 Текущая позиция *{player_name}*\nX: `{x:.1f}` Z: `{z:.1f}`"
                await query.edit_message_text(text, parse_mode='Markdown')

            elif action == "history":
                history = tracking_data['path_history'][-5:]  # Последние 5 позиций
                if not history:
                    text = "📜 История перемещений отсутствует"
                else:
                    text = "📜 Последние 5 позиций:\n" + "\n".join(
                        [f"X: {x:.1f}, Z: {z:.1f}" for (x, z), _ in history]
                    )
                await query.edit_message_text(text)

            elif action == "subscribe":
                user_id = query.from_user.id
                self.monitor.tracking_subscriptions[normalized_name].add(user_id)
                await query.edit_message_text(f"🔔 Вы подписались на обновления *{player_name}*", parse_mode='Markdown')

            else:
                await query.edit_message_text("⚠ Неизвестное действие")

        except Exception as e:
            logging.error(f"Ошибка обработки кнопки: {str(e)}", exc_info=True)
            await query.edit_message_text("❌ Произошла ошибка")

    async def unsubscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        # Обновление статуса в базе данных
        users = pd.read_csv(self.users_file)
        users.loc[users['user_id'] == user_id, 'subscribed'] = False
        users.to_csv(self.users_file, index=False)
        await update.message.reply_text("🔕 Вы отписались от уведомлений о зонах.")
    async def send_alert(self, alert: Alert):
        """Асинхронная отправка уведомлений с фильтрацией"""
        users = pd.read_csv(self.users_file)
        approved_users = users[users['approved']]

        emoji = {
            AlertLevel.INFO: 'ℹ',
            AlertLevel.WARNING: '⚠',
            AlertLevel.CRITICAL: '🚨'
        }.get(alert.level, '🔔')

        message = f"""
        {emoji} *{alert.source.capitalize()}*

        _{alert.message}_

        🕒 {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """

        # Отправка zone_intrusion всем approved пользователям
        if alert.source == "zone_intrusion":
            for user_id in approved_users['user_id']:
                try:
                    await self.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    logging.error(f"Ошибка отправки пользователю {user_id}: {str(e)}")

        # Остальные алерты отправляем только админу
        else:
            try:
                await self.bot.send_message(
                    chat_id=self.admin_id,
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logging.error(f"Ошибка отправки админу: {str(e)}")

    def sync_send_alert(self, alert: Alert):
        """Синхронная обертка для асинхронной отправки"""
        asyncio.run_coroutine_threadsafe(
            self.send_alert(alert),  # Передача корутины, а не результата
            self.loop
        )

    async def run_bot(self):
        try:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()

            # Бесконечный цикл с обработкой прерываний
            while True:
                await asyncio.sleep(3600)

        except asyncio.CancelledError:
            logging.info("Работа бота прервана")
        except Exception as e:
            logging.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        finally:
            await self.app.stop()
            await self.app.shutdown()

    def run(self):
        """Запуск бота с явным созданием цикла событий в потоке"""
        asyncio.set_event_loop(asyncio.new_event_loop())  # Создаем новый цикл
        self.app.run_polling()  # Запускаем бота в текущем цикле


class NoSos:
    def __init__(self, users_file='users.csv'):
        self.config = self.load_config()
        self.world_bounds = (
            self.config["world_bounds"]["xmin"],
            self.config["world_bounds"]["xmax"],
            self.config["world_bounds"]["zmin"],
            self.config["world_bounds"]["zmax"]
        )

        # 1. Инициализация TelegramBot ПЕРВОЙ
        self.telegram_bot = TelegramBot(self.config, self)
        self.start_telegram_bot()

        # 2. Инициализация остальных компонентов
        self.label_objects = []
        self.db_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.gui_queue = queue.Queue()
        self.alert_texts = []

        # 3. AlertManager получает ссылку после инициализации бота
        self.alert_manager = AlertManager(self)  # <-- self уже содержит telegram_bot
        self.setup_plot()
        self.init_data_structures()
        self.init_db()
        self.load_history()
        self.setup_alerts()
        self.start_data_thread()
        self.start_db_handler()
        self.load_translations()
        self.users_file = users_file
        self.admin_id = str(self.config['telegram']['chat_id'])
        self.activity_by_hour = defaultdict(int)  # Добавлено здесь

        self.security = SecurityManager(self.config)
        self.backup_system = BackupSystem()
        self.tracked_players = {}  # {player_name: (last_coords, update_time)}
        self.tracking_subscriptions = defaultdict(set)  # {player_name: {chat_ids}}
        self.analytics = AnalyticsEngine(self)

        # Запустить фоновые задачи
        self.start_background_tasks()

    def get_player_tracking_data(self, player_name: str) -> dict:
        try:
            current_pos, last_seen = self.tracked_players.get(player_name, ((None, None), 0.0))
            path_history = []

            # Фильтрация истории с проверкой типа данных
            for item in self.historical_data:
                if isinstance(item, tuple) and len(item) == 2:
                    pos, timestamp = item
                    if isinstance(pos, tuple) and len(pos) == 2 and pos[0] == player_name:
                        path_history.append((pos, timestamp))

            return {
                'current_pos': current_pos,
                'last_seen': last_seen,
                'path_history': path_history[-100:]  # Ограничение истории
            }
        except Exception as e:
            logging.error(f"Ошибка получения данных трекинга: {str(e)}")
            return {'current_pos': None, 'last_seen': 0.0, 'path_history': []}

    async def check_tracking_updates(self):
        """Проверка обновлений позиций с обработкой исключений"""
        while True:
            try:
                for player_name, chat_ids in self.tracking_subscriptions.items():
                    current_data = self.get_player_tracking_data(player_name)

                    if not current_data['path_history']:
                        continue

                    last_pos = current_data['path_history'][-1][0] if current_data['path_history'] else None
                    current_pos = current_data['current_pos']

                    if last_pos and current_pos and last_pos != current_pos:
                        for chat_id in chat_ids:
                            await self.telegram_bot.bot.send_message(
                                chat_id=chat_id,
                                text=f"🔔 *{player_name}* переместился\nНовая позиция: X: `{current_pos[0]:.1f}` Z: `{current_pos[1]:.1f}`",
                                parse_mode='Markdown'
                            )
                        self.last_reported_pos[player_name] = current_pos

                    await asyncio.sleep(60)
            except Exception as e:
                logging.error(f"Ошибка трекинга: {str(e)}")
                await asyncio.sleep(10)

    def start_background_tasks(self):
        async def _wrapper():
            # Создаем отдельный цикл для фоновых задач
            self.bg_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.bg_loop)

            # Запускаем задачи в новом цикле
            self.bg_loop.create_task(self.check_tracking_updates())
            self.bg_loop.create_task(self.backup_system.daily_backup())

            # Запускаем цикл в отдельном потоке
            await asyncio.sleep(0)

        threading.Thread(
            target=lambda: asyncio.run(_wrapper()),
            daemon=True,
            name="BackgroundTasks"
        ).start()

    def send_notifications(self, alert: Alert):
        """Используем синхронную обертку"""
        self.telegram_bot.sync_send_alert(alert)

    def start_telegram_bot(self):
        threading.Thread(
            target=self.telegram_bot.run,  # Теперь run() создает свой цикл
            daemon=True,
            name="TelegramBot"
        ).start()

    @staticmethod
    def load_config():
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        for zone in config['alerts']['zones']:
            zone['alert_level'] = AlertLevel[zone.get('alert_level', 'INFO')]

        if 'limits' in config['alerts']:
            config['alerts']['limits']['alert_level'] = AlertLevel[config['alerts']['limits']['alert_level']]

        return config

    def setup_plot(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(right=0.7, left=0.05)
        self.setup_controls()

    def setup_controls(self):
        self.player_list_ax = self.fig.add_axes([0.72, 0.25, 0.25, 0.70])
        self.player_list_ax.axis('off')
        self.player_list_text = self.player_list_ax.text(
            0.05, 0.95, "Обновление...",
            fontfamily='monospace',
            verticalalignment='top',
            color='white',
            fontsize=9
        )

    def init_data_structures(self):
        # Основные структуры данных
        self.current_data = []
        self.historical_data = deque(maxlen=self.config["heatmap"]["max_history"])
        self.data_lock = threading.Lock()
        self.stop_event = threading.Event()

        # Новая статистика
        self.activity_by_hour = defaultdict(int)
        self.player_time = defaultdict(float)
        self.zone_time = defaultdict(lambda: defaultdict(float))
        self.last_update_time = time.time()

        # Кэш для ускорения обработки
        self.zone_cache = {}

    def init_db(self):
        db_config = self.config['database']
        filename = db_config.get('filename', 'activity.db')
        self.conn = sqlite3.connect(filename, check_same_thread=False)
        self.create_tables()


    def start_db_handler(self):
        def db_handler():
            while not self.stop_event.is_set():
                try:
                    task = self.db_queue.get(timeout=1)
                    if task:
                        task()
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"DB handler error: {str(e)}")

        threading.Thread(target=db_handler, daemon=True).start()

    # В методе save_to_db заменяем работу с датами
    def save_to_db(self):
        def db_task():
            try:
                cursor = self.conn.cursor()
                now = datetime.now().date().isoformat()  # Преобразуем дату в строку ISO 8601
                current_hour = datetime.now().hour

                # Сохранение активности игроков
                for player, total_time in self.player_time.items():
                    cursor.execute('''
                        INSERT INTO activity (player, time, hour, date)
                        VALUES (?, ?, ?, ?)
                    ''', (player, total_time, current_hour, now))

                # Сохранение времени в зонах
                for zone, players in self.zone_time.items():
                    for player, time_spent in players.items():
                        cursor.execute('''
                            INSERT INTO zones (player, zone, time, date)
                            VALUES (?, ?, ?, ?)
                        ''', (player, zone, time_spent, now))

                self.conn.commit()
            except Exception as e:
                logging.error(f"Database error: {str(e)}")

        self.db_queue.put(db_task)

    def generate_stats_plot(self, update: Update):
        try:
            if not self.activity_by_hour:
                raise ValueError("Нет данных для статистики")
            plt.figure(figsize=(10, 5))
            plt.bar(self.activity_by_hour.keys(), self.activity_by_hour.values())
            plt.title("Активность игроков по часам")
            plt.xlabel("Час")
            plt.ylabel("Количество игроков")

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            async def send_plot():
                await self.telegram_bot.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=buf,
                    caption="📊 Статистика активности"
                )

            # Используем текущий цикл событий
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(send_plot(), loop)

        except Exception as e:
            logging.error(f"Ошибка генерации графика: {str(e)}")
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                update.message.reply_text(f"❌ Ошибка: {str(e)}"),
                loop
            )
    def create_tables(self):
        try:
            cursor = self.conn.cursor()
            # Убираем комментарии внутри SQL-запросов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity (
                    player TEXT,
                    time REAL,
                    hour INTEGER,
                    date TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS zones (
                    player TEXT,
                    zone TEXT,
                    time REAL,
                    date TEXT
                )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database error: {str(e)}")

    def load_translations(self):
        self.translations = {
            "en": {
                "welcome": "Welcome",
                "players_online": "Players Online (Overworld):",
                "alert": "Alert"
            },
            "ru": {
                "welcome": "Добро пожаловать",
                "players_online": "Онлайн игроки (Оверворлд):",
                "alert": "Оповещение"
            }
        }
        self.language = self.config.get("language", "ru")

    def setup_alerts(self):
        self.alert_manager.load_rules_from_config(self.config["alerts"])

    def start_data_thread(self):
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.data_worker())

        self.data_thread = threading.Thread(
            target=run_async_loop,
            daemon=True,
            name="DataWorkerThread"
        )
        self.data_thread.start()

    async def data_worker(self):
        while not self.stop_event.is_set():
            start_time = time.time()
            await self.fetch_and_process_data()  # Убедитесь, что это асинхронный метод
            elapsed = time.time() - start_time
            await asyncio.sleep(max(self.config["update_interval"] - elapsed, 0))

    def update_statistics(self):
        now = datetime.now()
        current_hour = now.hour

        with self.data_lock:
            # Обновление статистики активности
            self.activity_by_hour[current_hour] += len(self.current_data)

            # Обновление времени игроков
            time_diff = time.time() - self.last_update_time
            for player in self.current_data:
                name = player['name']
                self.player_time[name] += time_diff
                self.process_zone_time(player, time_diff)

            self.last_update_time = time.time()

    def process_zone_time(self, player, time_diff):
        pos = player.get("position", {})
        x, z = pos.get("x", 0), pos.get("z", 0)

        for zone in self.config["alerts"]["zones"]:
            zone_name = zone["name"]
            if self.is_in_zone(x, z, zone):
                self.zone_time[zone_name][player['name']] += time_diff
                break

    def is_in_zone(self, x, z, zone):
        cache_key = (x, z, zone['name'])
        if cache_key in self.zone_cache:
            return self.zone_cache[cache_key]

        in_zone = (
                zone["bounds"]["xmin"] <= x <= zone["bounds"]["xmax"] and
                zone["bounds"]["zmin"] <= z <= zone["bounds"]["zmax"]
        )
        self.zone_cache[cache_key] = in_zone
        return in_zone

    def recovery_protocol(self):
        """Восстановление после критических ошибок"""
        self.conn.close()
        self.init_db()
        self.load_history()
        self.setup_alerts()
        logging.info("System recovery completed")

    @alru_cache(maxsize=32)
    async def fetch_data(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_fetch_data)

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _sync_fetch_data(self):
        try:
            response = requests.get(self.config["players_url"], timeout=10)
            response.raise_for_status()
            return response.json().get("players", [])
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {str(e)}")
            raise

    @alru_cache(maxsize=32)
    async def fetch_and_process_data(self):
        try:
            all_players = await self.fetch_data()
            filtered_players = [p for p in all_players if not p.get('foreign', False)]

            with self.data_lock:
                self.current_data = filtered_players
                self.historical_data.extend(
                    [(p["position"]["x"], p["position"]["z"]) for p in filtered_players]
                )
                self.gui_update_queue.put(self.update_player_list_text)

            self.alert_manager.process_data({"players": filtered_players})
            self.fetch_data.cache_clear()  # Очистка кэша после успешной обработки

        except Exception as e:
            logging.error(f"Ошибка обработки данных: {str(e)}")

        except Exception as e:
            logging.error(f"Ошибка обработки данных: {str(e)}")

    def process_alerts(self):
        # Удаляем только несуществующие и успешно удаленные алерты
        current_texts = set(self.ax.texts)
        self.alert_texts = [
            alert_text for alert_text in self.alert_texts
            if alert_text in current_texts and not self.safe_remove(alert_text)
        ]

        # Добавляем новые алерты
        active_alerts = self.alert_manager.get_active_alerts()
        current_messages = {t.get_text() for t in self.alert_texts}

        for alert in active_alerts:
            alert_text = f"⚠ {alert.message} ⚠"
            if alert_text not in current_messages:
                self.show_alert(alert)
    def safe_remove(self, artist) -> bool:
        """Безопасное удаление художника с возвратом статуса успеха"""
        try:
            artist.remove()
            return True
        except ValueError:
            return False
        except Exception as e:
            logging.error(f"Ошибка удаления: {str(e)}")
            return False

    def show_alert(self, alert: Alert):
        try:
            color_map = {
                AlertLevel.INFO: '#FFFF00',
                AlertLevel.WARNING: '#FFA500',
                AlertLevel.CRITICAL: '#FF2222'
            }

            alert_text = self.ax.text(
                0.5,
                1.02,
                f"⚠ {alert.message} ⚠",
                transform=self.ax.transAxes,
                color=color_map[alert.level],
                fontsize=14,
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round',
                    facecolor='#330000',
                    edgecolor=color_map[alert.level],
                    linewidth=2
                ),
                zorder=100,
                alpha=0.9
            )
            self.alert_texts.append(alert_text)

            # Добавляем задачу удаления в очередь GUI
            def remove_task():
                if alert_text in self.ax.texts:
                    alert_text.remove()
                    self.alert_texts.remove(alert_text)
                    self.fig.canvas.draw_idle()

            self.gui_queue.put((remove_task, time.time() + 5.0))

        except Exception as e:
            logging.error(f"Ошибка отображения алерта: {str(e)}")

    def get_heatmap_bins(self):
        return [
            [self.world_bounds[0], self.world_bounds[1]],
            [self.world_bounds[2], self.world_bounds[3]]
        ]

    def draw_heatmap(self):
        try:
            if not self.historical_data or len(self.historical_data) < 10:
                return

            x, z = zip(*self.historical_data)
            x = [xi for xi in x if self.world_bounds[0] <= xi <= self.world_bounds[1]]
            z = [zi for zi in z if self.world_bounds[2] <= zi <= self.world_bounds[3]]

            if not x or not z:
                return

            self.ax.hist2d(
                x, z,
                bins=self.config["heatmap"]["bins"],
                cmap=self.config["heatmap"]["cmap"],
                alpha=self.config["heatmap"]["alpha"],
                zorder=-1,
                range=self.get_heatmap_bins(),
                density=True
            )
        except Exception as e:
            logging.error(f"Ошибка тепловой карты: {str(e)}")

    def update_plot(self, frame):
        try:
            now = time.time()
            while not self.gui_queue.empty():
                # Распаковываем три элемента: задача, аргументы, время
                task, args, execute_time = self.gui_queue.queue[0]
                if now >= execute_time:
                    self.gui_queue.get()
                    task(*args)  # Вызываем задачу с аргументами
                else:
                    break

            # Обработка задач БД в основном потоке
            while not self.db_queue.empty():
                task = self.db_queue.get()
                if task:
                    task()

            # Остальная логика отрисовки
            self.ax.clear()
            self.label_objects = []

            # Обновление данных интерфейса
            while not self.gui_update_queue.empty():
                update_func = self.gui_update_queue.get()
                update_func()

            if len(self.historical_data) >= 10:
                self.draw_heatmap()

            self.draw_zones()
            self.draw_players()
            self.setup_labels()
            self.update_player_list_text()
            self.process_alerts()

            # Перерисовка сохраненных алертов
            for alert_text in self.alert_texts:
                self.ax.add_artist(alert_text)

            return self.ax
        except Exception as e:
            logging.error(f"Ошибка обновления графика: {str(e)}")
            return self.ax
    def draw_players(self):
        with self.data_lock:
            filtered = self.current_data

        if filtered:
            x = [p["position"]["x"] for p in filtered]
            z = [p["position"]["z"] for p in filtered]

            self.ax.scatter(
                x, z,
                s=self.config["display"]["point_size"],
                c=self.config["display"]["point_color"],
                alpha=self.config["display"]["point_alpha"],
                edgecolors='none',
                zorder=10
            )

            label_config = self.config["display"]["labels"]
            for player in filtered:
                text = self.ax.annotate(
                    player['name'],
                    xy=(player['position']['x'], player['position']['z']),
                    xytext=(0, label_config["y_offset"]),
                    textcoords='offset points',
                    color=label_config["text_color"],
                    fontsize=label_config["font_size"],
                    ha='center',
                    alpha=0.9,
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor=label_config["bg_color"],
                        edgecolor='none',
                        alpha=label_config["bg_alpha"]
                    ),
                    zorder=11
                )
                self.label_objects.append(text)

            for player_name in self.tracking_subscriptions:
                if player_name in [p['name'] for p in self.current_data]:
                    # Подсветка отслеживаемых игроков
                    self.ax.scatter(
                        [p['position']['x'] for p in self.current_data if p['name'] == player_name],
                        [p['position']['z'] for p in self.current_data if p['name'] == player_name],
                        s=50,
                        color='#FF0000',
                        marker='X',
                        zorder=20
                    )

    def draw_zones(self):
        for zone in self.config["alerts"]["zones"]:
            xmin, xmax = zone["bounds"]["xmin"], zone["bounds"]["xmax"]
            zmin, zmax = zone["bounds"]["zmin"], zone["bounds"]["zmax"]

            edgecolor = 'gray' if zone.get("excluded") else 'red'
            linestyle = ':' if zone.get("excluded") else '--'

            self.ax.add_patch(plt.Rectangle(
                (xmin, zmin), xmax - xmin, zmax - zmin,
                fill=False,
                edgecolor=edgecolor,
                linestyle=linestyle,
                linewidth=1,
                zorder=9
            ))
            self.ax.text(
                xmin + 50, zmin + 50,
                f"{zone['name']} {'(excluded)' if zone.get('excluded') else ''}",
                color=edgecolor,
                zorder=11,
                fontsize=8
            )

    def setup_labels(self):
        self.ax.set_xlim(self.world_bounds[0], self.world_bounds[1])
        self.ax.set_ylim(self.world_bounds[2], self.world_bounds[3])
        self.ax.set_title(f"Карта активности игроков ({datetime.now().strftime('%H:%M:%S')})",
                          color='white', fontsize=12, pad=20)
        self.ax.grid(color='#404040', linestyle='--', linewidth=0.7)

    def update_player_list_text(self):
        with self.data_lock:
            players = sorted(self.current_data, key=lambda x: x['name'])
            text_lines = ["Онлайн игроки (Оверворлд):\n"]
            for player in players:
                x = int(player['position']['x'])
                z = int(player['position']['z'])
                text_lines.append(f"{player['name']: <20} X: {x: >6} Z: {z: >6}")

            self.player_list_text.set_text("\n".join(text_lines))

    def run(self):
        try:
            ani = FuncAnimation(
                self.fig, self.update_plot,
                interval=2000,
                cache_frame_data=False
            )
            plt.show()
        finally:
            self.shutdown()

    def translate(self, key):
        return self.translations[self.language].get(key, key)

    def set_theme(self, theme_name=None):
        theme_name = theme_name or self.config['themes'].get('default', 'dark')
        themes = {
            'dark': {'background': 'black', 'text': 'white'},
            'light': {'background': 'white', 'text': 'black'},
            'blue': {'background': '#002b36', 'text': '#839496'}
        }

        theme = themes.get(theme_name, themes['dark'])
        plt.style.use({'axes.facecolor': theme['background'],
                       'text.color': theme['text']})
        if hasattr(self, 'fig'):
            self.fig.set_facecolor(theme['background'])
            self.ax.set_facecolor(theme['background'])
            self.fig.canvas.draw_idle()

    def get_top_players(self, top_n=10):
        return sorted(self.player_time.items(),
                     key=lambda x: x[1], reverse=True)[:top_n]

    def get_zone_statistics(self):
        stats = {}
        for zone, players in self.zone_time.items():
            stats[zone] = sum(players.values())
        return stats


    def plot_activity_by_hour(self):
        hours = list(range(24))
        counts = [self.activity_by_hour.get(h, 0) for h in hours]

        plt.figure(figsize=(10, 6))
        plt.bar(hours, counts, color='blue')
        plt.xlabel(self.translate('hour'))
        plt.ylabel(self.translate('activity'))
        plt.title(self.translate('activity_by_hour'))
        plt.xticks(hours)
        plt.grid(True)
        plt.show()

    def export_data(self, format='csv'):
        try:
            if format == 'csv':
                self.export_to_csv()
            elif format == 'json':
                self.export_to_json()
            elif format == 'excel':
                self.export_to_excel()
            elif format == 'player_zone_time':
                self.export_player_zone_time()  # Новый метод для экспорта времени игроков в зонах
        except Exception as e:
            logging.error(f"Export error: {str(e)}")

    def export_player_zone_time(self):
        """Экспорт данных о времени, проведенном игроками в зонах."""
        try:
            player_zone_time = []
            for zone, players in self.zone_time.items():
                for player, time_spent in players.items():
                    player_zone_time.append([player, zone, round(time_spent, 2)])

            # Запись в CSV
            with open('player_zone_time.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Player', 'Zone', 'Time Spent (seconds)'])
                writer.writerows(player_zone_time)

            logging.info("Данные о времени игроков в зонах успешно экспортированы в player_zone_time.csv")
        except Exception as e:
            logging.error(f"Ошибка экспорта времени игроков в зонах: {str(e)}")

    def export_to_csv(self):
        # Экспорт активности игроков
        with open('player_activity.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Player', 'Total Time'])
            for player, time in self.player_time.items():
                writer.writerow([player, time])

        # Экспорт статистики по зонам
        with open('zone_stats.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Zone', 'Total Time'])
            for zone, time in self.get_zone_statistics().items():
                writer.writerow([zone, time])

    def export_to_json(self):
        data = {
            'player_activity': dict(self.player_time),
            'zone_stats': self.get_zone_statistics()
        }
        with open('stats.json', 'w') as f:
            json.dump(data, f, indent=2)

    def export_to_excel(self):
        df1 = pd.DataFrame(list(self.player_time.items()),
                           columns=['Player', 'Total Time'])
        df2 = pd.DataFrame(list(self.get_zone_statistics().items()),
                           columns=['Zone', 'Total Time'])

        with pd.ExcelWriter('stats.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Player Activity', index=False)
            df2.to_excel(writer, sheet_name='Zone Stats', index=False)

    def shutdown(self):
        # Остановка фоновых задач
        if hasattr(self, 'bg_loop') and self.bg_loop.is_running():
            self.bg_loop.call_soon_threadsafe(self.bg_loop.stop)

        # Остановка Telegram Bot
        if hasattr(self.telegram_bot, 'loop'):
            self.telegram_bot.loop.call_soon_threadsafe(
                self.telegram_bot.loop.stop
            )

        # Закрытие соединения с БД
        if hasattr(self, 'conn'):
            self.conn.close()

        # Дополнительные завершающие действия
        self.stop_event.set()
        plt.close('all')


    def save_history(self):
        try:
            with self.data_lock:
                history = list(self.historical_data)
            with open(self.config["heatmap"]["history_file"], 'wb') as f:
                pickle.dump(history, f)
            logging.info(f"Сохранено записей истории: {len(history)}")
        except Exception as e:
            logging.error(f"Ошибка сохранения истории: {str(e)}")

    def load_history(self):
        try:
            if os.path.exists(self.config["heatmap"]["history_file"]):
                with open(self.config["heatmap"]["history_file"], 'rb') as f:
                    history = pickle.load(f)
                with self.data_lock:
                    self.historical_data.extend(history)
                logging.info(f"Загружено записей истории: {len(history)}")
        except Exception as e:
            logging.error(f"Ошибка загрузки истории: {str(e)}")

if __name__ == "__main__":
    monitor = NoSos()
    monitor.run()
