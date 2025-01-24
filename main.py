import requests
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons
from datetime import datetime
import threading
import time
import logging
import queue
import unicodedata
from collections import deque
import os
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('monitor.log'), logging.StreamHandler()]
)


class SafeMonitor:
    def __init__(self):
        self.config = self.load_config()
        self.alert_queue = queue.Queue()
        self.gui_update_queue = queue.Queue()
        self.setup_plot()
        self.init_data_structures()
        self.load_history()
        self.start_data_thread()

    def load_config(self):
        return {
            "players_url": "https://map.serverchichi.online/maps/world/live/players.json",
            "world_bounds": {"xmin": -10145, "xmax": 10145, "zmin": -10078, "zmax": 10078},
            "update_interval": 15,
            "min_request_interval": 5,
            "display": {
                "point_size": 25,
                "point_color": "#00FF00",
                "point_alpha": 0.7,
                "labels": {
                    "font_size": 8,
                    "text_color": "#FFFF00",
                    "bg_color": "#000000",
                    "bg_alpha": 0.5,
                    "y_offset": 8
                }
            },
            "alerts": {
                "zones": [
                    {"name": "Спавн", "bounds": {"xmin": -500, "xmax": 500, "zmin": -500, "zmax": 500}},
                    {
                        "name": "Сосмарк",
                        "bounds": {"xmin": 4124, "xmax": 5291, "zmin": -3465, "zmax": -2225},
                        "allowed_players": [
                            "K1zik", "Kwakich", "_DeN41k_", "Bruno_Hempf", "Sanplay11",
                            "Panzergrenadier", "napacoJlbka", "Amoliper", "Chestz",
                            "BotEnot", "_mixailchert_", "VuLo4ka", "Aphgust", "Sir_Bred",
                            "Timondarck", "italianopelmen", "KomJys", "BaBiEdA",
                            "poshelyanaher", "Peridotik", "_ChesTer_aD", "RockVandal"
                        ]
                    }
                ],
                "max_players": 50
            },
            "heatmap": {
                "bins": 50,
                "cmap": "hot",
                "alpha": 0.3,
                "show": False,
                "mode": "history",
                "max_history": 10 ** 6,
                "history_file": "heatmap_history.pkl"
            }
        }

    def setup_plot(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(right=0.7, left=0.05)
        self.setup_controls()

    def setup_controls(self):
        # Основные элементы управления
        self.player_list_ax = self.fig.add_axes([0.72, 0.15, 0.25, 0.80])
        self.player_list_ax.axis('off')
        self.player_list_text = self.player_list_ax.text(
            0.05, 0.95, "Обновление...", fontfamily='monospace',
            verticalalignment='top', color='white'
        )

        # Чекбокс для игроков
        self.checkbox_ax = self.fig.add_axes([0.72, 0.10, 0.25, 0.04])
        self.player_checkboxes = CheckButtons(self.checkbox_ax, [], [])
        self.player_checkboxes.on_clicked(self.update_filter)

        # Управление heatmap
        self.heatmap_control_ax = self.fig.add_axes([0.72, 0.01, 0.25, 0.08])
        self.heatmap_control_ax.axis('off')

        # Чекбокс отображения heatmap
        self.heatmap_checkbox = CheckButtons(
            self.fig.add_axes([0.73, 0.05, 0.1, 0.04]),
            ['Тепловая карта'],
            [self.config["heatmap"]["show"]]
        )
        self.heatmap_cid = self.heatmap_checkbox.on_clicked(self.toggle_heatmap)

        # Чекбокс режима heatmap
        self.heatmap_mode_checkbox = CheckButtons(
            self.fig.add_axes([0.73, 0.01, 0.2, 0.04]),
            ['Режим истории'],
            [self.config["heatmap"]["mode"] == "history"]
        )
        self.heatmap_mode_cid = self.heatmap_mode_checkbox.on_clicked(self.toggle_heatmap_mode)

    def toggle_heatmap(self, label):
        if isinstance(label, str):
            self.config["heatmap"]["show"] = not self.config["heatmap"]["show"]
            self.heatmap_checkbox.disconnect(self.heatmap_cid)
            self.heatmap_checkbox.set_active([0] if self.config["heatmap"]["show"] else [])
            self.heatmap_cid = self.heatmap_checkbox.on_clicked(self.toggle_heatmap)

    def toggle_heatmap_mode(self, label):
        if isinstance(label, str):
            new_mode = "history" if self.config["heatmap"]["mode"] == "current" else "current"
            self.config["heatmap"]["mode"] = new_mode
            self.heatmap_mode_checkbox.disconnect(self.heatmap_mode_cid)
            self.heatmap_mode_checkbox.set_active([0] if new_mode == "history" else [])
            self.heatmap_mode_cid = self.heatmap_mode_checkbox.on_clicked(self.toggle_heatmap_mode)

    def init_data_structures(self):
        self.current_data = []
        self.historical_data = deque(maxlen=self.config["heatmap"]["max_history"])
        self.players_list = set()
        self.selected_players = set()
        self.data_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.alerts = []
        self.setup_alerts()
        self.label_objects = []

    def setup_alerts(self):
        self.alerts.extend([{
            "type": "zone",
            "zone": zone["name"],
            "bounds": zone["bounds"],
            "triggered": False
        } for zone in self.config["alerts"]["zones"]])

    def start_data_thread(self):
        self.data_thread = threading.Thread(target=self.data_worker, daemon=True)
        self.data_thread.start()

    def data_worker(self):
        while not self.stop_event.is_set():
            try:
                start_time = time.perf_counter()
                self.fetch_and_process_data()
                self.check_alerts()

                # Рассчёт оставшегося времени с защитой от слишком частых запросов
                elapsed = time.perf_counter() - start_time
                sleep_time = max(
                    self.config["min_request_interval"],
                    self.config["update_interval"] - elapsed
                )

                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Data thread error: {str(e)}")
                time.sleep(self.config["update_interval"])  # Защитная задержка при ошибках

    def fetch_and_process_data(self):
        try:
            response = requests.get(self.config["players_url"], timeout=10)
            if response.status_code == 200:
                all_players = response.json().get("players", [])
                filtered_players = [p for p in all_players if not p.get('foreign', False)]
                with self.data_lock:
                    self.current_data = filtered_players
                    self.historical_data.extend(
                        [(p["position"]["x"], p["position"]["z"]) for p in filtered_players]
                    )
                    self.gui_update_queue.put(lambda: self.update_players_list(filtered_players))
        except Exception as e:
            logging.error(f"Fetch error: {str(e)}")

    def save_history(self):
        try:
            with self.data_lock:
                history = list(self.historical_data)
            with open(self.config["heatmap"]["history_file"], 'wb') as f:
                pickle.dump(history, f)
            logging.info(f"Сохранено {len(history)} записей истории")
        except Exception as e:
            logging.error(f"Ошибка сохранения истории: {str(e)}")

    def load_history(self):
        try:
            if os.path.exists(self.config["heatmap"]["history_file"]):
                with open(self.config["heatmap"]["history_file"], 'rb') as f:
                    history = pickle.load(f)
                with self.data_lock:
                    self.historical_data.extend(history)
                logging.info(f"Загружено {len(history)} исторических записей")
        except Exception as e:
            logging.error(f"Ошибка загрузки истории: {str(e)}")

    def update_players_list(self, players):
        new_players = {p["name"] for p in players} - self.players_list
        if new_players:
            self.players_list.update(new_players)
            for p in new_players:
                self.player_checkboxes.labels.append(plt.Text(0, 0, p))
            self.fig.canvas.draw_idle()

    def update_filter(self, label):
        if label in self.selected_players:
            self.selected_players.remove(label)
        else:
            self.selected_players.add(label)

    def check_alerts(self):
        with self.data_lock:
            current_players = {p["name"] for p in self.current_data}
            positions = {p["name"]: (p["position"]["x"], p["position"]["z"]) for p in self.current_data}

        alerts_to_show = []
        for alert in self.alerts:
            if alert["type"] == "zone":
                result = self.check_zone_alert(alert, positions)
                if result: alerts_to_show.append(result)

        if len(self.current_data) > self.config["alerts"]["max_players"]:
            alerts_to_show.append(f"Players limit exceeded: {len(self.current_data)}")

        for alert in alerts_to_show:
            self.alert_queue.put(alert)

    def normalize_name(self, name: str) -> str:
        cleaned = unicodedata.normalize('NFKD', name)
        cleaned = cleaned.encode('ascii', 'ignore').decode()
        return cleaned.strip().lower()

    def check_zone_alert(self, alert, positions):
        zone_name = alert["zone"]
        try:
            zone_config = next(z for z in self.config["alerts"]["zones"] if z["name"] == zone_name)
        except StopIteration:
            logging.error(f"Конфигурация зоны не найдена: {zone_name}")
            return None

        allowed_players = set(self.normalize_name(p) for p in zone_config.get("allowed_players", []))
        logging.info(f"Проверка зоны {zone_name}. Игнорируемые игроки: {allowed_players}")

        for name, (x, z) in positions.items():
            clean_name = self.normalize_name(name)

            bounds = alert["bounds"]
            in_zone = (
                    bounds["xmin"] <= x <= bounds["xmax"] and
                    bounds["zmin"] <= z <= bounds["zmax"]
            )

            if not in_zone:
                continue

            if zone_name == "Сосмарк":
                if clean_name in allowed_players:
                    logging.info(f"Игрок {name} ({clean_name}) в белом списке, пропуск")
                    continue

            if not alert["triggered"]:
                alert["triggered"] = True
                logging.warning(f"Срабатывание алерта для {name} в {zone_name}")
                return f"Player {name} in {alert['zone']}"

        alert["triggered"] = False
        return None

    def process_alerts(self):
        while not self.alert_queue.empty():
            alert = self.alert_queue.get()
            logging.warning(f"ALERT: {alert}")
            self.show_alert(alert)

    def show_alert(self, message):
        self.ax.annotate(
            f"! {message} !",
            xy=(0.5, 1.02),
            xycoords='axes fraction',
            color='red',
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )
        plt.draw()
        plt.pause(3)
        self.ax.texts[-1].remove()

    def draw_heatmap(self):
        try:
            if not self.historical_data and not self.current_data:
                return

            heatmap_config = self.config["heatmap"]

            if heatmap_config["mode"] == "history":
                data = list(self.historical_data)
            else:
                data = [(p["position"]["x"], p["position"]["z"])
                        for p in self.current_data]

            if not data:
                return

            x, z = zip(*data)

            self.ax.hist2d(
                x, z,
                bins=heatmap_config["bins"],
                cmap=heatmap_config["cmap"],
                alpha=heatmap_config["alpha"],
                zorder=-1,
                range=[
                    [self.config["world_bounds"]["xmin"], self.config["world_bounds"]["xmax"]],
                    [self.config["world_bounds"]["zmin"], self.config["world_bounds"]["zmax"]]
                ]
            )
        except Exception as e:
            logging.error(f"Heatmap error: {str(e)}")

    def update_plot(self, frame):
        try:
            while not self.gui_update_queue.empty():
                update_func = self.gui_update_queue.get()
                update_func()

            self.ax.clear()

            if self.config["heatmap"]["show"]:
                self.draw_heatmap()

            self.draw_players()
            self.draw_zones()
            self.setup_labels()
            self.update_player_list_text()
            self.process_alerts()
            return self.ax
        except Exception as e:
            logging.error(f"Plot error: {str(e)}")
            return self.ax

    def draw_players(self):
        with self.data_lock:
            filtered = [p for p in self.current_data
                        if not self.selected_players or p["name"] in self.selected_players]

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
            self.ax.add_patch(plt.Rectangle(
                (xmin, zmin), xmax - xmin, zmax - zmin,
                fill=False, edgecolor='red', linestyle='--', linewidth=1,
                zorder=10
            ))
            self.ax.text(
                xmin + 50, zmin + 50, zone["name"],
                color='red', zorder=11
            )

    def setup_labels(self):
        self.ax.set_xlim(self.config["world_bounds"]["xmin"], self.config["world_bounds"]["xmax"])
        self.ax.set_ylim(self.config["world_bounds"]["zmin"], self.config["world_bounds"]["zmax"])
        self.ax.set_title(f"Player Activity Map ({datetime.now().strftime('%H:%M:%S')})", color='white')
        self.ax.grid(color='#30363d', linestyle='--')

    def update_player_list_text(self):
        with self.data_lock:
            players = sorted(self.current_data, key=lambda x: x['name'])
            text_lines = ["Online Players (Overworld):\n"]
            for player in players:
                x = int(player['position']['x'])
                z = int(player['position']['z'])
                text_lines.append(f"{player['name']: <20} X: {x: >6} Z: {z: >6}")

            self.player_list_text.set_text("\n".join(text_lines))

    def run(self):
        try:
            # Увеличиваем интервал обновления GUI до 2 секунд
            ani = FuncAnimation(
                self.fig, self.update_plot,
                interval=2000,  # Было 1000
                cache_frame_data=False
            )
            plt.show()
        finally:
            self.shutdown()

    def shutdown(self):
        self.stop_event.set()
        self.save_history()  # Сохранение истории перед выходом
        plt.close('all')


if __name__ == "__main__":
    monitor = SafeMonitor()
    monitor.run()
