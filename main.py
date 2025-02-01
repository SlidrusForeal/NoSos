import csv
import json
import logging
import os
import pickle
import asyncpg
import queue
import sqlite3
import time
import unicodedata
import re
import numpy as np
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yaml
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import threading
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest
import asyncio
from io import BytesIO
from PIL import Image

request = HTTPXRequest()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://postgres:IgCDeaSMmvIUsvwUrcUAwygSBvPmSylG@postgres.railway.internal:5432/railway"


class AlertLevel(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


async def init_db():
    return await asyncpg.connect(DATABASE_URL)


async def create_tables():
    conn = await init_db()
    try:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id BIGINT PRIMARY KEY,
                username TEXT,
                approved BOOLEAN DEFAULT FALSE,
                subscribed BOOLEAN DEFAULT FALSE
            )
        ''')

        await conn.execute('''
            CREATE TABLE IF NOT EXISTS activity (
                id SERIAL PRIMARY KEY,
                player TEXT,
                time FLOAT,
                hour INTEGER,
                date DATE
            )
        ''')

        await conn.execute('''
            CREATE TABLE IF NOT EXISTS zones (
                id SERIAL PRIMARY KEY,
                player TEXT,
                zone TEXT,
                time FLOAT,
                date DATE
            )
        ''')
    finally:
        await conn.close()

@dataclass
class Alert:
    message: str
    level: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    cooldown: float = 60


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

                # Рассчет расстояния и времени
                distance = self._calculate_distance(last_pos, current_pos)
                time_diff = current_time - last_time

                # Проверка на аномалии
                if time_diff > 0:
                    speed = distance / time_diff
                    if speed > self.max_speed:
                        if distance > self.teleport_threshold:
                            alert = self._create_teleport_alert(player, distance)
                        else:
                            alert = self._create_speed_alert(player, speed)

                        if self._should_trigger(alert.message):
                            alerts.append(alert)
                            self._update_cooldown(alert.message)

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
    def _calculate_distance(pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

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
    def __init__(self):
        self.rules: List[BaseAlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=1000)

    def get_active_alerts(self) -> List[Alert]:
        """Возвращает список активных алертов"""
        return list(self.active_alerts.values())

    @lru_cache(maxsize=100)
    def _generate_alert_id(self, source: str, message: str) -> str:
        return f"{source}_{hash(message)}"

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
                alerts = rule.check_conditions(data)
                for alert in alerts:
                    alert_id = self._generate_alert_id(alert.source, alert.message)
                    if alert_id not in self.active_alerts:
                        new_alerts.append(alert)
                        self.active_alerts[alert_id] = alert
            except Exception as e:
                logging.error(f"Ошибка в правиле {rule.__class__.__name__}: {str(e)}")

        for alert in new_alerts:
            alert_id = self._generate_alert_id(alert.source, alert.message)
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            monitor.send_notifications(alert)

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


class AnalyticsEngine:
    def __init__(self, monitor):
        self.monitor = monitor
        self.config = monitor.config.get("analytics", {})
        self.max_speed = self.config.get("max_speed", 50)
        self.teleport_threshold = self.config.get("teleport_threshold", 100)

    def detect_anomalies(self, player_data: dict) -> str:
        anomalies = []
        if player_data.get("speed", 0) > self.max_speed:
            anomalies.append("Высокая скорость перемещения")
        if player_data.get("distance", 0) > self.teleport_threshold:
            anomalies.append("Подозрительная телепортация")
        return " | ".join(anomalies) if anomalies else "Норма"

    def generate_heatmap_report(self) -> str:
        # zone_time может быть как словарь вида {zone: {player: time}} или другой – здесь пример реализации
        zone_activity = {}
        for zone, players in self.monitor.zone_time.items():
            if isinstance(players, dict):
                zone_activity[zone] = sum(players.values())
            else:
                zone_activity[zone] = players
        sorted_zones = sorted(zone_activity.items(), key=lambda x: x[1], reverse=True)[:5]
        report_lines = [f"• {zone}: {time // 60} минут" for zone, time in sorted_zones]
        return "🔥 Топ-5 активных зон:\n" + "\n".join(report_lines)

    def generate_player_report(self, player_name: str) -> str:
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
    def __init__(self, config, monitor):
        self.monitor = monitor
        self.config = config
        self.admin_id = str(config['telegram']['chat_id'])
        self.bot = Bot(token=config['telegram']['token'])
        self.app = ApplicationBuilder().token(config['telegram']['token']).build()
        self._register_handlers()

        # Инициализируем модуль аналитики
        self.analytics = AnalyticsEngine(monitor)

    def _register_handlers(self):
        handlers = [
            CommandHandler("start", self.start),
            CommandHandler("help", self.help_command),
            CommandHandler("unsubscribe", self.unsubscribe),
            CommandHandler("approve", self.approve_user),
            CommandHandler("users", self.list_users),
            CommandHandler("send", self.send_message_command),
            CommandHandler("caramel_pain", self.caramel_pain_command),
            CommandHandler("history", self.history),
            CommandHandler("subscribe", self.subscribe),
            # Новые команды для аналитики (только для админа)
            CommandHandler("anomalies", self.anomalies),
            CommandHandler("heatmap", self.heatmap),
            CommandHandler("player_report", self.player_report),
            CallbackQueryHandler(self.handle_callback),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message),
        ]
        for handler in handlers:
            self.app.add_handler(handler)

    async def _check_admin(self, update: Update) -> bool:
        user_id = str(update.effective_user.id)
        if not self.monitor.security.is_admin(user_id):
            await update.message.reply_text(
                "⛔ Ты адекатная? А ничо тот факт что ты не администратор бота и у тебя жижа за 50 рублей купленая у ашота. \n жди докс короче")
            return False
        return True

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        user_id = user.id
        try:
            username = user.username or ""
            avatar = None
            user_profile_photos = await context.bot.get_user_profile_photos(user_id, limit=1)
            if user_profile_photos.total_count > 0:
                avatar = user_profile_photos.photos[0][-1]
                conn = await init_db()
                try:
                    exists = await conn.fetchval(
                        'SELECT 1 FROM users WHERE user_id = $1', user.id
                    )
                    if not exists:
                        await conn.execute('''
                            INSERT INTO users (user_id, username, approved, subscribed)
                            VALUES ($1, $2, $3, $4)
                        ''', user.id, user.username, False, False)
                finally:
                    await conn.close()

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
            logging.error(f"Ошибка в команде /start: {str(e)}", exc_info=True)
            await update.message.reply_text(
                "⚠️ *Произошла ошибка при обработке запроса*",
                parse_mode='Markdown'
            )

    async def approve_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update):
            await update.message.reply_text("❌ У вас нет прав для выполнения этой команды.")
            return

        if not context.args or not context.args[0].isdigit():
            await update.message.reply_text("❌ Используйте: /approve <ID пользователя>")
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
                text="✅ <b>Ваш аккаунт одобрен!</b>\n\n"
                     "Теперь вам доступны команды:\n"
                     "/help - Справка\n"
                     "/subscribe - Подписка\n"
                     "/unsubscribe - Отписка\n"
                     "/history - Топ игроков\n",
                parse_mode=ParseMode.HTML
            )

        except Exception as e:
            logging.error(f"Error in approve_user: {e}")
            await update.message.reply_text("Произошла ошибка при одобрении пользователя.")

    async def list_users(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update):
            return

        users = pd.read_csv(self.users_file)
        text = "📋 *Список пользователей:*\n"
        for _, row in users.iterrows():
            status = "✅ Одобрен" if row['approved'] else "⏳ Ожидает"
            text += f"🆔 `{row['user_id']}` | 👤 {row['username'] or 'N/A'} | {status}\n"
        await update.message.reply_text(text, parse_mode='Markdown')

    async def send_message_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_admin(update):
            return

        if not context.args or len(context.args) < 2:
            await update.message.reply_text("❌ Формат: /send <ID> <сообщение>")
            return

        user_id, message = context.args[0], " ".join(context.args[1:])
        try:
            await self.bot.send_message(chat_id=user_id, text=f"🔔 Сообщение от админа:\n{message}")
            await update.message.reply_text("✅ Сообщение отправлено")
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")

    async def caramel_pain_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        responses = ["Кто такие мышериоты?", "La-Li-Lu-Le-Lo", "Shin Sei Moku Roku"]
        await update.message.reply_text(f"🔐 {random.choice(responses)}")

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
                    text="✅ <b>Одобрено!</b>\n\n"
                         "Доступные команды:\n"
                         "/help - Помощь\n"
                         "/subscribe - Уведомления\n"
                         "/unsubscribe - Отмена\n"
                         "/history - Активность\n"
                         "/caramel_pain - Тайна",
                    parse_mode=ParseMode.HTML
                )
                if query.message and query.message.text:
                    await query.edit_message_text(text=f"Пользователь {user_id} одобрен")
                else:
                    await query.message.reply_text(f"Пользователь {user_id} одобрен")

            elif action == "reject":
                # Логика отклонения заявки
                with self.users_lock:
                    users = pd.read_csv(self.users_file)
                    if int(user_id) in users['user_id'].values:
                        # Удаляем пользователя из списка заявок
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
        user_id = update.effective_user.id
        conn = await init_db()
        try:
            await conn.execute('''
                UPDATE users SET subscribed = FALSE WHERE user_id = $1
            ''', user_id)
            await update.message.reply_text("🔕 Вы отписались от уведомлений!")
        finally:
            await conn.close()

    async def subscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        conn = await init_db()
        try:
            await conn.execute('''
                UPDATE users SET subscribed = TRUE WHERE user_id = $1
            ''', user_id)
            await update.message.reply_text("🔔 Вы подписались на уведомления!")
        finally:
            await conn.close()

    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        users = pd.read_csv(self.users_file)

        if user_id not in users['user_id'].astype(str).values:
            await update.message.reply_text("❌ Вы не зарегистрированы.")
            return

        history = self.monitor.get_top_players()
        response = "📜 Топ активных игроков:\n"
        for player, time in history:
            response += f"{player}: {time // 60} минут\n"

        await update.message.reply_text(response)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        is_admin = self.monitor.security.is_admin(user_id)

        help_text = "🛠 <b>Доступные команды</b>\n\n"
        help_text += "<b>Основные:</b>\n"
        help_text += "/help - Справка\n/subscribe - Подписка\n/unsubscribe - Отписка\n/history - Топ игроков\n\n"

        if is_admin:
            help_text += "<b>Админ:</b>\n"
            help_text += "/users - Список\n/approve - Одобрить\n/send - Сообщение\n/anomalies - Проверка\n/heatmap - Зоны\n/player_report - Отчёт\n"

        await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("ℹ Используйте /help для списка команд")

    async def anomalies(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Команда: /anomalies <скорость> <дистанция>
        Проверяет аномалии по данным игрока.
        """
        if not await self._check_admin(update):
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
        """
        Команда: /heatmap
        Генерирует отчёт по топ-5 активным зонам.
        """
        if not await self._check_admin(update):
            return

        result = self.analytics.generate_heatmap_report()
        await update.message.reply_text(result)

    async def player_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Команда: /player_report <имя игрока>
        Отчёт по времени игрока и активности по зонам.
        """
        if not await self._check_admin(update):
            return

        if not context.args:
            await update.message.reply_text("❌ Используйте: /player_report <имя игрока>")
            return

        player_name = " ".join(context.args)
        result = self.analytics.generate_player_report(player_name)
        await update.message.reply_text(result, parse_mode='Markdown')

    async def run(self):
        await self.app.run_polling()

    @staticmethod
    @lru_cache(maxsize=1000)
    def _normalize_name(name: str) -> str:
        """Нормализует имя для callback_data"""
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
        return re.sub(r'[^a-zA-Z0-9_]', '', name).lower()


class NoSos:
    def __init__(self):
        self.config = self.load_config()
        self.loop = asyncio.get_event_loop()
        self.security = SecurityManager(self.config)
        self.telegram_bot = TelegramBot(self.config, self)
        self.world_bounds = (
            self.config["world_bounds"]["xmin"],
            self.config["world_bounds"]["xmax"],
            self.config["world_bounds"]["zmin"],
            self.config["world_bounds"]["zmax"]
        )
        self.config.setdefault('database', {'filename': 'activity.db'})
        self.config.setdefault('language', 'ru')
        self.config.setdefault('themes', {'default': 'dark'})

        self.label_objects = []
        self.db_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.gui_queue = queue.Queue()
        self.alert_texts = []

        self.alert_manager = AlertManager()
        self.setup_plot()
        self.init_data_structures()
        self.init_db()
        self.load_history()
        self.setup_alerts()
        self.start_data_thread()
        self.start_db_handler()
        self.load_translations()
        asyncio.run(create_tables())

    def send_notifications(self, alert: Alert):
        asyncio.run(self._async_send_alert(alert))

    def _run_loop(self, loop):
        """Функция для запуска event loop в отдельном потоке"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _async_send_alert(self, alert: Alert):
        try:
            users = pd.read_csv(self.telegram_bot.users_file)
            approved_users = users[users['approved'] & users['subscribed']]

            # Определяем целевую аудиторию
            if alert.source == "zone_intrusion":
                recipients = approved_users['user_id'].tolist()
            else:
                # Только админы из конфига
                recipients = self.config['security']['admins']

            # Формируем сообщение
            message = (
                f"🚨 *{alert.source.upper()}* 🚨\n"
                f"_Игрок {alert.metadata.get('player', 'Unknown')}_\n"
                f"🕒 {alert.timestamp.strftime('%Y-%m-%d %H:%M')}"
            )

            # Отправка
            for user_id in recipients:
                await self.telegram_bot.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode='Markdown'
                )
                await asyncio.sleep(0.5)

        except Exception as e:
            logging.error(f"Ошибка отправки алерта: {str(e)}")


    @staticmethod
    def load_config():
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    async def save_activity(self, player_data: dict):
        conn = await init_db()
        try:
            await conn.execute('''
                INSERT INTO activity (player, time, hour, date)
                VALUES ($1, $2, $3, $4)
            ''',
            player_data['name'],
            player_data['time'],
            datetime.now().hour,
            datetime.now().date())
        finally:
            await conn.close()

    def setup_plot(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(
            figsize=(16, 10)
        )
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(right=0.7, left=0.05)
        self.setup_controls()

        self.ax = self.fig.add_subplot(111)

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
        self.data_thread = threading.Thread(
            target=self.data_worker,
            daemon=True,
            name="DataWorkerThread"
        )
        self.data_thread.start()

    def data_worker(self):
        while not self.stop_event.is_set():
            try:
                start_time = time.time()
                self.fetch_and_process_data()
                self.update_statistics()
                self.save_to_db()

                elapsed = time.time() - start_time
                sleep_time = max(
                    self.config["min_request_interval"],
                    self.config["update_interval"] - elapsed
                )
                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Data worker error: {str(e)}")

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

    @lru_cache(maxsize=32)
    def fetch_data(self):
        try:
            response = requests.get(self.config["players_url"], timeout=10)
            if response.status_code == 200:
                return response.json().get("players", [])
            return []
        except Exception as e:
            logging.error(f"Ошибка получения данных: {str(e)}")
            return []

    def fetch_and_process_data(self):
        try:
            all_players = self.fetch_data()
            filtered_players = [p for p in all_players if not p.get('foreign', False)]

            with self.data_lock:
                self.current_data = filtered_players
                self.historical_data.extend(
                    [(p["position"]["x"], p["position"]["z"]) for p in filtered_players]
                )
                self.gui_update_queue.put(self.update_player_list_text)

            # Передача данных в AlertManager для проверки алертов
            self.alert_manager.process_data({"players": filtered_players})  # <-- Добавлено

            self.fetch_data.cache_clear()

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
            # Обработка задач GUI из очереди
            now = time.time()
            while not self.gui_queue.empty():
                task, execute_time = self.gui_queue.queue[0]
                if now >= execute_time:
                    self.gui_queue.get()
                    task()
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
            head_cache = {}
            label_config = self.config["display"]["labels"]

            # Загрузка всех изображений заранее
            for player in filtered:
                username = player['name']
                if username not in head_cache:
                    try:
                        response = requests.get(
                            f"https://serverchichi.online/api/getHead/{username}.png",
                            timeout=3
                        )
                        img = Image.open(BytesIO(response.content))
                        head_cache[username] = img
                    except Exception as e:
                        head_cache[username] = None

            # Отрисовка элементов в правильном порядке
            for player in filtered:
                x = player['position']['x']
                z = player['position']['z']
                username = player['name']
                img = head_cache.get(username)

                # 1. Отрисовка голов/точек (нижний слой)
                if img:
                    img_array = np.array(img)
                    imagebox = OffsetImage(img_array, zoom=0.15)
                    ab = AnnotationBbox(
                        imagebox,
                        (x, z),
                        frameon=False,
                        pad=0,
                        zorder=10  # Головы под метками, но над фоном
                    )
                    self.ax.add_artist(ab)
                else:
                    self.ax.scatter(
                        x, z,
                        s=self.config["display"]["point_size"],
                        c=self.config["display"]["point_color"],
                        alpha=self.config["display"]["point_alpha"],
                        edgecolors='none',
                        zorder=10
                    )

                # 2. Отрисовка меток (верхний слой)
                text = self.ax.annotate(
                    username,
                    xy=(x, z),
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
                    zorder=20  # Метки поверх всех элементов
                )
                self.label_objects.append(text)

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
        self.ax.set_zorder(10)
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
            ani.save('animation.gif', writer='pillow')  # Сохранить анимацию в файл
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
        # Дожидаемся завершения задач БД
        while not self.db_queue.empty():
            task = self.db_queue.get()
            if task:
                task()

        # Закрываем соединение
        if hasattr(self, 'conn'):
            self.conn.close()

        # Остальная логика завершения...
        self.stop_event.set()
        plt.close('all')
        self.export_data()

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
