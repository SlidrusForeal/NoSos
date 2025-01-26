import requests
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import threading
import time
import logging
import queue
import unicodedata
from collections import deque
import os
import pickle
from functools import lru_cache
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, List
import yaml

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


class BaseAlertRule(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cooldown = config.get("cooldown", 60)
        self.last_triggered: Dict[str, float] = {}
        self.alert_level = config.get("alert_level", AlertLevel.WARNING)

    @abstractmethod
    def check_conditions(self, data: Dict) -> List[Alert]:
        pass

    def _should_trigger(self, identifier: str) -> bool:
        return (time.time() - self.last_triggered.get(identifier, 0)) > self.cooldown

    def _update_cooldown(self, identifier: str):
        self.last_triggered[identifier] = time.time()


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
            }
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
            "player_limit": PlayerCountRule
        }

        for rule_config in config.get("zones", []):
            self.rules.append(rule_registry["zone_intrusion"](rule_config))

        if "limits" in config:
            self.rules.append(rule_registry["player_limit"](config["limits"]))

    def process_data(self, data: Dict):
        new_alerts = []
        for rule in self.rules:
            try:
                new_alerts.extend(rule.check_conditions(data))
            except Exception as e:
                logging.error(f"Ошибка в правиле {rule.__class__.__name__}: {str(e)}")

        for alert in new_alerts:
            alert_id = self._generate_alert_id(alert.source, alert.message)
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

        self._clean_expired_alerts()

    def _clean_expired_alerts(self):
        expire_time = time.time() - 3600
        to_remove = [aid for aid, alert in self.active_alerts.items()
                    if alert.timestamp.timestamp() < expire_time]
        for aid in to_remove:
            del self.active_alerts[aid]

    def get_alert_history(self, limit=50) -> List[Alert]:
        return list(self.alert_history)[-limit:]


class NoSos:
    def __init__(self):
        self.config = self.load_config()
        self.world_bounds = (
            self.config["world_bounds"]["xmin"],
            self.config["world_bounds"]["xmax"],
            self.config["world_bounds"]["zmin"],
            self.config["world_bounds"]["zmax"]
        )
        self.gui_update_queue = queue.Queue()
        self.alert_manager = AlertManager()
        self.setup_plot()
        self.init_data_structures()
        self.load_history()
        self.setup_alerts()
        self.start_data_thread()

    def load_config(self):
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
        self.current_data = []
        self.historical_data = deque(maxlen=self.config["heatmap"]["max_history"])
        self.data_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.label_objects = []

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
                start_time = time.perf_counter()
                self.fetch_and_process_data()

                with self.data_lock:
                    data = {"players": self.current_data}
                    self.alert_manager.process_data(data)

                elapsed = time.perf_counter() - start_time
                sleep_time = max(
                    self.config["min_request_interval"],
                    self.config["update_interval"] - elapsed
                )
                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Ошибка потока данных: {str(e)}")
                time.sleep(self.config["update_interval"])

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

            self.fetch_data.cache_clear()

        except Exception as e:
            logging.error(f"Ошибка обработки данных: {str(e)}")

    def process_alerts(self):
        for alert in self.alert_manager.get_active_alerts():
            self.show_alert(alert)
            logging.warning(f"АКТИВНЫЙ АЛЕРТ: {alert.message}")

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

            def remove_alert():
                if alert_text in self.ax.texts:
                    alert_text.remove()
                    self.fig.canvas.draw_idle()

            threading.Timer(5.0, remove_alert).start()
            self.fig.canvas.draw_idle()

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
            while not self.gui_update_queue.empty():
                update_func = self.gui_update_queue.get()
                update_func()

            self.ax.clear()

            for label in self.label_objects:
                if label in self.ax.texts:
                    self.ax.texts.remove(label)
            self.label_objects.clear()

            if len(self.historical_data) >= 10:
                self.draw_heatmap()

            self.draw_zones()
            self.draw_players()
            self.setup_labels()
            self.update_player_list_text()
            self.process_alerts()

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

    def shutdown(self):
        self.stop_event.set()
        self.save_history()
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
