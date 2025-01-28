import csv
import json
import logging
import os
import pickle
import queue
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import requests
import unicodedata
import yaml
from matplotlib.animation import FuncAnimation
from telegram import Bot, Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
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
        return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()

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
        self.triggered_alerts = set()  # Храним уникальные ID сработавших алертов

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
                new_alerts.extend(rule.check_conditions(data))
            except Exception as e:
                logging.error(f"Ошибка в правиле {rule.__class__.__name__}: {str(e)}")

        for alert in new_alerts:
            alert_id = self._generate_alert_id(alert.source, alert.message)
            if alert_id not in self.triggered_alerts: # Проверяем, не срабатывал ли алерт ранее
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                self.triggered_alerts.add(alert_id)  # Добавляем ID в множество сработавших
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


class TelegramNotifier:
    def __init__(self, token: str, monitor_instance=None):
        self.bot = Bot(token=token)
        self.config_file = "bot_users.json"
        self.monitor = monitor_instance  # Ссылка на монитор
        self.registered_users = set()  # Зарегистрированные пользователи (chat_id)

        self._load_users()
        self.application = Application.builder().token(token).build()

        # Регистрация обработчиков
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("alerts", self.handle_alerts))
        self.application.add_handler(CallbackQueryHandler(self.handle_button))

        # Запуск бота
        self.application.run_polling()

    def _load_users(self):
        """Загрузка списка пользователей из файла"""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                self.registered_users = set(json.load(f))

    def _save_users(self):
        """Сохранение списка пользователей в файл"""
        with open(self.config_file, "w") as f:
            json.dump(list(self.registered_users), f)

    async def handle_start(self, update: Update, context):
        """Обработчик команды /start"""
        user = update.effective_user
        chat_id = update.effective_chat.id

        if chat_id not in self.registered_users:
            self.registered_users.add(chat_id)
            self._save_users()
            logging.info(f"Новый пользователь зарегистрирован: {user.username} (chat_id: {chat_id})")

        await update.message.reply_text(
            "✅ Добро пожаловать! Вы успешно зарегистрированы.\n"
            "Вы будете получать уведомления.",
            parse_mode="Markdown"
        )

    async def send_alert(self, alert):
        """Отправка уведомления всем пользователям"""
        if not self.registered_users:
            logging.warning("Нет зарегистрированных пользователей для отправки уведомлений.")
            return

        message = self._format_alert_message(alert)
        reply_markup = self._create_alert_keyboard(alert)

        for chat_id in self.registered_users:
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode="Markdown",
                    reply_markup=reply_markup,
                    disable_web_page_preview=True
                )
            except Exception as e:
                logging.error(f"Ошибка отправки сообщения для chat_id {chat_id}: {e}")

    def _format_alert_message(self, alert) -> str:
        """Форматирование сообщения об алерте"""
        message = (
            f"🚨 *{alert.source.upper()} Alert* 🚨\n"
            f"• Уровень: `{alert.level.name}`\n"
            f"• Сообщение: {alert.message}\n"
            f"• Время: {alert.timestamp.strftime('%H:%M:%S')}\n"
        )

        if alert.metadata:
            metadata = "\n".join([f"• {k}: `{v}`" for k, v in alert.metadata.items()])
            message += f"\n*Детали:*\n{metadata}"

        return message

    def _create_alert_keyboard(self, alert) -> InlineKeyboardMarkup:
        """Создание интерактивной клавиатуры для алерта"""
        buttons = [
            [
                InlineKeyboardButton("🔄 Обновить", callback_data="refresh"),
                InlineKeyboardButton("❌ Закрыть", callback_data="close")
            ],
            [
                InlineKeyboardButton(
                    "📊 Статистика",
                    url=f"http://your-monitor-domain.com/stats/{alert.source}"
                )
            ]
        ]
        return InlineKeyboardMarkup(buttons)

    async def handle_alerts(self, update: Update, context):
        """Обработчик команды /alerts"""
        if not self.monitor:
            await update.message.reply_text("⚠️ Система мониторинга не доступна.")
            return

        alerts = self.monitor.alert_manager.get_alert_history(10)
        if not alerts:
            await update.message.reply_text("ℹ️ Нет активных оповещений.")
            return

        response = ["*Последние 10 оповещений:*\n"]
        for i, alert in enumerate(reversed(alerts), 1):
            response.append(
                f"{i}. [{alert.level.name}] {alert.message} "
                f"({alert.timestamp.strftime('%H:%M:%S')})"
            )

        await update.message.reply_text(
            "\n".join(response),
            parse_mode="Markdown"
        )

    async def handle_button(self, update: Update, context):
        """Обработчик инлайн-кнопок"""
        query = update.callback_query
        await query.answer()

        if query.data == "refresh":
            await self._handle_refresh(query)
        elif query.data == "close":
            await self._handle_close(query)

    async def _handle_refresh(self, query):
        """Обновление сообщения"""
        try:
            await query.edit_message_text(
                text=f"{query.message.text}\n\n🔄 Обновлено: {datetime.now().strftime('%H:%M:%S')}",
                parse_mode="Markdown"
            )
        except Exception as e:
            logging.error(f"Ошибка обновления сообщения: {str(e)}")

    async def _handle_close(self, query):
        """Удаление сообщения"""
        try:
            await query.delete_message()
        except Exception as e:
            logging.error(f"Ошибка удаления сообщения: {str(e)}")

class NoSos:
    def __init__(self):
        self.config = self.load_config()
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
        self.init_telegram()


    def init_telegram(self):
        bot = TelegramNotifier("7998680701:AAGl29f22cMIcC6qXVmMszwjYREbMaBztRo")

    def send_notifications(self, alert: Alert):
        if hasattr(self, 'tg_notifier'):
            self.tg_notifier.send_alert(alert)
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
