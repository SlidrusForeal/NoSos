import os
import sqlite3
import queue
import threading
import time
import logging
from collections import defaultdict, deque
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import tempfile
import re
import asyncio

from alerts.manager import AlertManager
from bot.telegram_bot import TelegramBot
from security.security_manager import SecurityManager
from db.database import get_connection, create_tables
from utils.helpers import clean_html_tags

class NOSOS:
    def __init__(self, config):
        self.config = config
        self.security = SecurityManager(config)
        self.telegram_bot = TelegramBot(config, self)
        self.bot = self.telegram_bot.bot
        self.world_bounds = (
            config["world_bounds"]["xmin"],
            config["world_bounds"]["xmax"],
            config["world_bounds"]["zmin"],
            config["world_bounds"]["zmax"]
        )
        self.conn = get_connection(config['database'].get('filename', 'activity.db'))
        create_tables(self.conn)
        self.label_objects = []
        self.db_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.gui_queue = queue.Queue()
        self.temp_db_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.start_temp_db_handler()
        self.alert_manager = AlertManager(self)
        self.setup_plot()
        self.init_data_structures()
        self.load_history()
        self.setup_alerts()
        self.start_data_thread()
        self.start_db_handler()
        self.load_translations()
        self.init_temp_db()
        threading.Thread(target=self.telegram_bot.run, daemon=True).start()
        self.start_cleanup_thread()
        self.sent_alerts_cache = set()
        self.alert_cache_lock = threading.Lock()

    def start_cleanup_thread(self):
        def cleanup_worker():
            while not self.stop_event.is_set():
                try:
                    cursor = self.temp_conn.cursor()
                    cursor.execute('''
                        DELETE FROM player_movements 
                        WHERE timestamp < datetime('now', '-30 days')
                    ''')
                    self.temp_conn.commit()
                    time.sleep(3600)
                except Exception as e:
                    logging.error(f"Cleanup error: {e}")
        threading.Thread(target=cleanup_worker, daemon=True).start()

    def send_notifications(self, alert):
        try:
            loop = self.telegram_bot.app._loop
            asyncio.run_coroutine_threadsafe(self._async_send_alert(alert), loop)
        except Exception as e:
            logging.error(f"Ошибка отправки уведомления: {e}")

    async def _async_send_alert(self, alert):
        try:
            if not alert.message.strip():
                logging.error("Пустое сообщение алерта")
                return
            alert_key = hash(alert.message)
            with self.alert_cache_lock:
                if alert_key in self.sent_alerts_cache:
                    logging.info(f"Алерт уже был отправлен: {alert.message}")
                    return
                self.sent_alerts_cache.add(alert_key)
            if alert.source == "movement_anomaly":
                message = (
                    f"🚨 *Превышение скорости* 🚨\n"
                    f"Игрок: _{alert.metadata['player']}_\n"
                    f"Скорость: `{alert.metadata['speed']}` блоков/сек\n"
                    f"🕒 {alert.timestamp.strftime('%H:%M:%S')}"
                )
            elif alert.source == "zone_intrusion":
                message = (
                    f"🚨 *Вторжение в зону {alert.metadata['zone']}* 🚨\n"
                    f"👥 Игроки ({alert.metadata['count']}): {', '.join(alert.metadata['players'])}\n"
                    f"🕒 {alert.timestamp.strftime('%H:%M:%S')}"
                )
            else:
                message = (
                    f"🚨 *Телепортация / смертьь* 🚨\n"
                    f"Источник: {alert.source.upper()}\n"
                    f"Игрок: {alert.metadata.get('player', 'Неизвестно')}\n"
                    f"🕒 {alert.timestamp.strftime('%H:%M:%S')}"
                )
            from telegram.helpers import escape_markdown
            safe_message = escape_markdown(message, version=2)
            if alert.level.name == "CRITICAL":
                await self.bot.send_message(
                    chat_id=self.config["telegram"]["chat_id"],
                    text=safe_message,
                    parse_mode='MarkdownV2'
                )
            else:
                users = pd.read_csv("users.csv")
                approved_users = users[users['approved'] & users['subscribed']]
                for user_id in approved_users['user_id']:
                    try:
                        await self.bot.send_message(
                            chat_id=str(user_id),
                            text=safe_message,
                            parse_mode='MarkdownV2'
                        )
                        await asyncio.sleep(0.3)
                    except Exception as e:
                        logging.error(f"Ошибка отправки пользователю {user_id}: {e}")
        except Exception as e:
            logging.error(f"Ошибка отправки алерта: {e}", exc_info=True)

    def init_temp_db(self):
        self.temp_conn = sqlite3.connect('data.db', check_same_thread=False)
        self.create_temp_tables()

    def create_temp_tables(self):
        try:
            cursor = self.temp_conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS player_movements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player TEXT NOT NULL,
                    x REAL NOT NULL,
                    z REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracking (
                    player TEXT PRIMARY KEY,
                    last_x REAL,
                    last_z REAL,
                    last_update DATETIME
                )
            ''')
            self.temp_conn.commit()
        except Exception as e:
            logging.error(f"Temp DB error: {e}")

    def start_temp_db_handler(self):
        def handler():
            while not self.stop_event.is_set():
                try:
                    task = self.temp_db_queue.get(timeout=1)
                    if task:
                        task()
                except Exception:
                    continue
        threading.Thread(target=handler, daemon=True).start()

    def process_player_movements(self, players):
        tracked_messages = []
        try:
            cursor = self.temp_conn.cursor()
            current_time = datetime.now().isoformat()
            for player in players:
                name = player['name']
                x = player['position']['x']
                z = player['position']['z']
                cursor.execute('''
                    INSERT INTO player_movements (player, x, z)
                    VALUES (?, ?, ?)
                ''', (name, x, z))
                last_pos = self.telegram_bot.player_history.get(name.lower(), {"x": None, "z": None})
                if (x != last_pos["x"]) or (z != last_pos["z"]):
                    self.telegram_bot.player_history[name.lower()] = {"x": x, "z": z}
                    with self.telegram_bot.track_lock:
                        subscribers = self.telegram_bot.tracked_players.get(name.lower(), set())
                        if subscribers:
                            msg = (
                                f"📡 *{name}* переместился\n"
                                f"📍 X: `{int(x)}` Z: `{int(z)}`\n"
                                f"🕒 {datetime.now().strftime('%H:%M:%S')}"
                            )
                            tracked_messages.append((msg, subscribers))
                cursor.execute('''
                    INSERT OR REPLACE INTO tracking 
                    (player, last_x, last_z, last_update)
                    VALUES (?, ?, ?, ?)
                ''', (name, x, z, current_time))
            self.temp_conn.commit()
            if tracked_messages:
                self.send_track_notifications(tracked_messages)
        except Exception as e:
            logging.error(f"Ошибка обработки перемещений: {e}")
            self.temp_conn.rollback()

    async def _async_send_track_notification(self, message, user_ids):
        try:
            for user_id in user_ids:
                await self.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode='Markdown'
                )
                await asyncio.sleep(0.3)
        except Exception as e:
            logging.error(f"Ошибка отправки трек-уведомления: {e}")

    def send_track_notifications(self, messages):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True).start()
        for msg, user_ids in messages:
            asyncio.run_coroutine_threadsafe(
                self._async_send_track_notification(msg, user_ids),
                loop
            )

    def get_player_history(self, player_name, limit=100):
        try:
            cursor = self.temp_conn.cursor()
            cursor.execute('''
                SELECT x, z, timestamp 
                FROM player_movements 
                WHERE player = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (player_name, limit))
            return cursor.fetchall()
        except Exception as e:
            logging.error(f"Get history error: {e}")
            return []

    def get_last_position(self, player_name):
        try:
            cursor = self.temp_conn.cursor()
            cursor.execute('''
                SELECT last_x, last_z, last_update 
                FROM tracking 
                WHERE player = ?
            ''', (player_name,))
            return cursor.fetchone()
        except Exception as e:
            logging.error(f"Get position error: {e}")
            return None

    def load_history(self):
        # Имплементация загрузки истории, если требуется
        pass

    def setup_plot(self):
        matplotlib.use('Qt5Agg')
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10), num="NOSOS")
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(right=0.7, left=0.05)
        self.setup_controls()
        try:
            from PyQt5 import QtGui
            manager = plt.get_current_fig_manager()
            if manager and hasattr(manager, 'window'):
                manager.window.setWindowIcon(QtGui.QIcon("icon.ico"))
        except Exception as e:
            logging.error(f"Ошибка установки иконки: {e}")

    def setup_controls(self):
        self.player_list_ax = self.fig.add_axes([0.72, 0.25, 0.25, 0.70])
        self.player_list_ax.axis('off')
        self.player_list_text = self.player_list_ax.text(0.05, 0.95, "Обновление...", fontfamily='monospace', verticalalignment='top', color='white', fontsize=9)

    def init_data_structures(self):
        self.current_data = []
        self.historical_data = deque(maxlen=self.config["heatmap"]["max_history"])
        self.data_lock = threading.Lock()
        self.activity_by_hour = defaultdict(int)
        self.player_time = defaultdict(float)
        self.zone_time = defaultdict(lambda: defaultdict(float))
        self.last_update_time = time.time()
        self.zone_cache = {}

    def start_db_handler(self):
        def db_handler():
            while not self.stop_event.is_set():
                try:
                    task = self.db_queue.get(timeout=1)
                    if task:
                        task()
                except Exception:
                    continue
        threading.Thread(target=db_handler, daemon=True).start()

    def save_to_db(self):
        def db_task():
            try:
                cursor = self.conn.cursor()
                now = datetime.now().isoformat()
                current_hour = datetime.now().hour
                for player, total_time in self.player_time.items():
                    cursor.execute('''
                        INSERT INTO activity (player, time, hour, date)
                        VALUES (?, ?, ?, ?)
                    ''', (player, total_time, current_hour, now))
                for zone, players in self.zone_time.items():
                    for player, time_spent in players.items():
                        cursor.execute('''
                            INSERT INTO zones (player, zone, time, date)
                            VALUES (?, ?, ?, ?)
                        ''', (player, zone, time_spent, now))
                self.conn.commit()
            except Exception as e:
                logging.error(f"Database error: {e}")
        self.db_queue.put(db_task)

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
        self.data_thread = threading.Thread(target=self.data_worker, daemon=True, name="DataWorkerThread")
        self.data_thread.start()

    def data_worker(self):
        while not self.stop_event.is_set():
            try:
                start_time = time.time()
                self.fetch_and_process_data()
                self.update_statistics()
                self.save_to_db()
                elapsed = time.time() - start_time
                sleep_time = max(self.config["min_request_interval"], self.config["update_interval"] - elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Data worker error: {e}")

    def update_statistics(self):
        now = datetime.now()
        current_hour = now.hour
        with self.data_lock:
            self.activity_by_hour[current_hour] += len(self.current_data)
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
        in_zone = (zone["bounds"]["xmin"] <= x <= zone["bounds"]["xmax"] and zone["bounds"]["zmin"] <= z <= zone["bounds"]["zmax"])
        self.zone_cache[cache_key] = in_zone
        return in_zone

    def get_zone_name_by_coordinates(self, x, z):
        for zone in self.config["alerts"]["zones"]:
            if self.is_in_zone(x, z, zone):
                return zone["name"]
        return None

    def fetch_data(self):
        try:
            response = requests.get(self.config["players_url"], timeout=10)
            if response.status_code == 200:
                return response.json().get("players", [])
            return []
        except Exception as e:
            logging.error(f"Ошибка получения данных: {e}")
            return []

    def fetch_and_process_data(self):
        import requests
        try:
            all_players = []
            # Получаем ссылки из конфигурации
            players_urls = self.config.get("players_urls", {})
            for zone, url in players_urls.items():
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    zone_players = response.json().get("players", [])
                    # Добавляем метку источника, чтобы различать данные
                    for player in zone_players:
                        player["zone_source"] = zone
                    all_players.extend(zone_players)
            # Фильтрация данных по миру, если требуется, и добавление метки
            filtered_players = [
                {
                    "name": p["name"],
                    "position": {"x": p["x"], "z": p["z"]},
                    "uuid": p["uuid"],
                    "world": p["world"],
                    "zone_source": p.get("zone_source", "unknown")
                }
                for p in all_players if p["world"] == "minecraft_overworld"
            ]
            with self.data_lock:
                self.current_data = filtered_players
                self.historical_data.extend([(p["position"]["x"], p["position"]["z"]) for p in filtered_players])
                self.gui_update_queue.put(self.update_player_list_text)
            self.alert_manager.process_data({"players": filtered_players})
            self.process_player_movements(filtered_players)
        except Exception as e:
            logging.error(f"Ошибка обработки данных: {e}")

    def draw_heatmap(self):
        # Имплементация отрисовки тепловой карты
        pass

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
                self.ax.annotate(
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
        self.ax.set_xlim((self.world_bounds[0], self.world_bounds[1]))
        self.ax.set_ylim((self.world_bounds[2], self.world_bounds[3]))
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

    def update_plot(self, frame):
        try:
            now = time.time()
            while not self.gui_queue.empty():
                task, execute_time = self.gui_queue.queue[0]
                if now >= execute_time:
                    self.gui_queue.get()
                    task()
                else:
                    break
            while not self.db_queue.empty():
                task = self.db_queue.get()
                if task:
                    task()
            self.ax.clear()
            while not self.gui_update_queue.empty():
                update_func = self.gui_update_queue.get()
                update_func()
            if len(self.historical_data) >= 10:
                self.draw_heatmap()
            self.draw_zones()
            self.draw_players()
            self.setup_labels()
            self.update_player_list_text()
            return self.ax
        except Exception as e:
            logging.error(f"Ошибка обновления графика: {e}")
            return self.ax

    def run(self):
        from matplotlib.animation import FuncAnimation
        ani = FuncAnimation(self.fig, self.update_plot, interval=2000, cache_frame_data=False)
        plt.show()
        self.shutdown()

    def shutdown(self):
        self.stop_event.set()
        self.conn.close()
        self.temp_conn.close()

    def get_top_players(self, top_n=10):
        return sorted(self.player_time.items(), key=lambda x: x[1], reverse=True)[:top_n]
