# Импорт необходимых библиотек
import requests  # Для выполнения HTTP-запросов
import matplotlib.pyplot as plt  # Для создания визуализации
from matplotlib.animation import FuncAnimation  # Для анимации графика
from matplotlib.widgets import CheckButtons  # Для чекбоксов управления
from datetime import datetime  # Для работы с временными метками
import threading  # Для работы с потоками
import time  # Для работы с задержками
import logging  # Для логирования событий
import queue  # Для организации очередей между потоками
import unicodedata  # Для нормализации строк
from collections import deque  # Для работы с историей позиций
import os  # Для работы с файловой системой
import pickle  # Для сохранения и загрузки истории

# Настройка системы логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),  # Запись логов в файл
        logging.StreamHandler()  # Вывод логов в консоль
    ]
)

class NoSos:
    def __init__(self):
        """Инициализация основного класса монитора."""
        self.config = self.load_config()  # Загрузка конфигурации
        self.alert_queue = queue.Queue()  # Очередь для оповещений
        self.gui_update_queue = queue.Queue()  # Очередь для обновления GUI
        self.setup_plot()  # Настройка графического интерфейса
        self.init_data_structures()  # Инициализация структур данных
        self.load_history()  # Загрузка истории тепловой карты
        self.start_data_thread()  # Запуск потока сбора данных

    def load_config(self):
        """Загрузка конфигурации приложения."""
        return {
            "players_url": "https://map.serverchichi.online/maps/world/live/players.json",
            "world_bounds": {  # Границы игрового мира
                "xmin": -10145, 
                "xmax": 10145, 
                "zmin": -10078, 
                "zmax": 10078
            },
            "update_interval": 15,  # Интервал обновления данных (секунды)
            "min_request_interval": 5,  # Минимальный интервал между запросами
            "display": {  # Настройки отображения
                "point_size": 25,  # Размер точек игроков
                "point_color": "#00FF00",  # Цвет точек
                "point_alpha": 0.7,  # Прозрачность точек
                "labels": {  # Настройки подписей
                    "font_size": 8,
                    "text_color": "#FFFF00",
                    "bg_color": "#000000",
                    "bg_alpha": 0.5,
                    "y_offset": 8  # Смещение подписи относительно точки
                }
            },
            "alerts": {  # Настройки системы оповещений
                "zones": [  # Контролируемые зоны
                    {
                        "name": "Спавн",
                        "bounds": {"xmin": -500, "xmax": 500, "zmin": -500, "zmax": 500}
                    },
                    {
                        "name": "Сосмарк",
                        "bounds": {"xmin": 4124, "xmax": 5291, "zmin": -3465, "zmax": -2225},
                        "allowed_players": [  # Белый список игроков
                            "K1zik", "Kwakich", "_DeN41k_", "Bruno_Hempf", "Sanplay11",
                            "Panzergrenadier", "napacoJlbka", "Amoliper", "Chestz",
                            "BotEnot", "_mixailchert_", "VuLo4ka", "Aphgust", "Sir_Bred",
                            "Timondarck", "italianopelmen", "KomJys", "BaBiEdA",
                            "poshelyanaher", "Peridotik", "_ChesTer_aD", "RockVandal"
                        ]
                    }
                ],
                "max_players": 50  # Максимальное количество игроков без предупреждения
            },
            "heatmap": {  # Настройки тепловой карты
                "bins": 50,  # Количество секторов
                "cmap": "hot",  # Цветовая схема
                "alpha": 0.3,  # Прозрачность
                "show": False,  # Показывать по умолчанию
                "mode": "history",  # Режим отображения (history/current)
                "max_history": 10**6,  # Максимальный размер истории
                "history_file": "heatmap_history.pkl"  # Файл для сохранения истории
            }
        }

    def setup_plot(self):
        """Настройка графического интерфейса."""
        plt.style.use('dark_background')  # Темная тема
        self.fig = plt.figure(figsize=(16, 10))  # Создание фигуры
        self.ax = self.fig.add_subplot(111)  # Основные оси для карты
        self.fig.subplots_adjust(right=0.7, left=0.05)  # Настройка расположения
        self.setup_controls()  # Добавление элементов управления

    def setup_controls(self):
        """Создание элементов управления."""
        # Область для списка игроков
        self.player_list_ax = self.fig.add_axes([0.72, 0.15, 0.25, 0.80])
        self.player_list_ax.axis('off')
        self.player_list_text = self.player_list_ax.text(
            0.05, 0.95, "Обновление...", 
            fontfamily='monospace',
            verticalalignment='top', 
            color='white'
        )

        # Чекбоксы для фильтрации игроков
        self.checkbox_ax = self.fig.add_axes([0.72, 0.10, 0.25, 0.04])
        self.player_checkboxes = CheckButtons(self.checkbox_ax, [], [])
        self.player_checkboxes.on_clicked(self.update_filter)

        # Элементы управления тепловой картой
        self.heatmap_control_ax = self.fig.add_axes([0.72, 0.01, 0.25, 0.08])
        self.heatmap_control_ax.axis('off')

        # Чекбокс отображения тепловой карты
        self.heatmap_checkbox = CheckButtons(
            self.fig.add_axes([0.73, 0.05, 0.1, 0.04]),
            ['Тепловая карта'],
            [self.config["heatmap"]["show"]]
        )
        self.heatmap_cid = self.heatmap_checkbox.on_clicked(self.toggle_heatmap)

        # Переключатель режима тепловой карты
        self.heatmap_mode_checkbox = CheckButtons(
            self.fig.add_axes([0.73, 0.01, 0.2, 0.04]),
            ['Режим истории'],
            [self.config["heatmap"]["mode"] == "history"]
        )
        self.heatmap_mode_cid = self.heatmap_mode_checkbox.on_clicked(self.toggle_heatmap_mode)

    def toggle_heatmap(self, label):
        """Переключение отображения тепловой карты."""
        self.config["heatmap"]["show"] = not self.config["heatmap"]["show"]
        self.heatmap_checkbox.set_active([0] if self.config["heatmap"]["show"] else [])

    def toggle_heatmap_mode(self, label):
        """Переключение режима тепловой карты между текущими и историческими данными."""
        new_mode = "history" if self.config["heatmap"]["mode"] == "current" else "current"
        self.config["heatmap"]["mode"] = new_mode
        self.heatmap_mode_checkbox.set_active([0] if new_mode == "history" else [])

    def init_data_structures(self):
        """Инициализация структур данных."""
        self.current_data = []  # Текущие данные о позициях
        self.historical_data = deque(maxlen=self.config["heatmap"]["max_history"])  # История позиций
        self.players_list = set()  # Список всех игроков
        self.selected_players = set()  # Выбранные для отображения игроки
        self.data_lock = threading.Lock()  # Блокировка для потокобезопасности
        self.stop_event = threading.Event()  # Событие для остановки потока
        self.alerts = []  # Список активных оповещений
        self.setup_alerts()  # Инициализация системы оповещений
        self.label_objects = []  # Графические элементы подписей

    def setup_alerts(self):
        """Инициализация системы оповещений для зон."""
        for zone in self.config["alerts"]["zones"]:
            self.alerts.append({
                "type": "zone",
                "zone": zone["name"],
                "bounds": zone["bounds"],
                "triggered": False  # Флаг срабатывания
            })

    def start_data_thread(self):
        """Запуск потока для сбора данных."""
        self.data_thread = threading.Thread(target=self.data_worker, daemon=True)
        self.data_thread.start()

    def data_worker(self):
        """Основной цикл потока сбора данных."""
        while not self.stop_event.is_set():
            try:
                start_time = time.perf_counter()
                self.fetch_and_process_data()  # Получение и обработка данных
                self.check_alerts()  # Проверка условий для оповещений

                # Расчет времени до следующего обновления
                elapsed = time.perf_counter() - start_time
                sleep_time = max(
                    self.config["min_request_interval"],
                    self.config["update_interval"] - elapsed
                )
                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Ошибка в потоке данных: {str(e)}")
                time.sleep(self.config["update_interval"])

    def fetch_and_process_data(self):
        """Получение и обработка данных с сервера."""
        try:
            response = requests.get(self.config["players_url"], timeout=10)
            if response.status_code == 200:
                # Фильтрация данных
                all_players = response.json().get("players", [])
                filtered_players = [p for p in all_players if not p.get('foreign', False)]
                
                with self.data_lock:
                    self.current_data = filtered_players
                    # Сохранение в историю
                    self.historical_data.extend(
                        [(p["position"]["x"], p["position"]["z"]) for p in filtered_players]
                    )
                    # Обновление GUI
                    self.gui_update_queue.put(lambda: self.update_players_list(filtered_players))
        except Exception as e:
            logging.error(f"Ошибка получения данных: {str(e)}")

    def save_history(self):
        """Сохранение истории позиций в файл."""
        try:
            with self.data_lock:
                history = list(self.historical_data)
            with open(self.config["heatmap"]["history_file"], 'wb') as f:
                pickle.dump(history, f)
            logging.info(f"Сохранено {len(history)} записей истории")
        except Exception as e:
            logging.error(f"Ошибка сохранения истории: {str(e)}")

    def load_history(self):
        """Загрузка истории позиций из файла."""
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
        """
        Обновление списка игроков в боковой панели.
        Добавляет новых игроков в чекбоксы фильтрации.
        """
        # Находим новых игроков, которых еще нет в текущем списке
        new_players = {p["name"] for p in players} - self.players_list
        
        if new_players:
            # Обновляем основной список игроков
            self.players_list.update(new_players)
            
            # Добавляем новых игроков в элементы управления
            for p in new_players:
                self.player_checkboxes.labels.append(plt.Text(0, 0, p))
            
            # Перерисовываем интерфейс
            self.fig.canvas.draw_idle()

    def update_filter(self, label):
        """
        Обработчик изменения состояния чекбоксов.
        Обновляет список выбранных для отображения игроков.
        """
        if label in self.selected_players:
            self.selected_players.remove(label)
        else:
            self.selected_players.add(label)

    def check_alerts(self):
        """
        Проверка всех условий для срабатывания оповещений.
        """
        with self.data_lock:
            # Получаем текущие данные о позициях игроков
            current_players = {p["name"] for p in self.current_data}
            positions = {p["name"]: (p["position"]["x"], p["position"]["z"]) for p in self.current_data}

        alerts_to_show = []
        # Проверяем все зарегистрированные оповещения
        for alert in self.alerts:
            if alert["type"] == "zone":
                result = self.check_zone_alert(alert, positions)
                if result: 
                    alerts_to_show.append(result)

        # Проверка на превышение максимального числа игроков
        if len(self.current_data) > self.config["alerts"]["max_players"]:
            alerts_to_show.append(f"Players limit exceeded: {len(self.current_data)}")

        # Помещаем все оповещения в очередь обработки
        for alert in alerts_to_show:
            self.alert_queue.put(alert)

    def normalize_name(self, name: str) -> str:
        """
        Нормализация имени игрока для сравнения.
        Удаляет диакритические знаки и приводит к нижнему регистру.
        """
        cleaned = unicodedata.normalize('NFKD', name)
        cleaned = cleaned.encode('ascii', 'ignore').decode()
        return cleaned.strip().lower()

    def check_zone_alert(self, alert, positions):
        """
        Проверка конкретной зоны на наличие неавторизованных игроков.
        Возвращает сообщение об оповещении при обнаружении нарушения.
        """
        zone_name = alert["zone"]
        try:
            # Получаем конфигурацию зоны из настроек
            zone_config = next(z for z in self.config["alerts"]["zones"] if z["name"] == zone_name)
        except StopIteration:
            logging.error(f"Конфигурация зоны не найдена: {zone_name}")
            return None

        # Подготавливаем белый список игроков для этой зоны
        allowed_players = set(self.normalize_name(p) for p in zone_config.get("allowed_players", []))
        logging.info(f"Проверка зоны {zone_name}. Игнорируемые игроки: {allowed_players}")

        # Проверяем всех игроков в текущих позициях
        for name, (x, z) in positions.items():
            clean_name = self.normalize_name(name)

            # Получаем границы зоны
            bounds = alert["bounds"]
            in_zone = (
                bounds["xmin"] <= x <= bounds["xmax"] and
                bounds["zmin"] <= z <= bounds["zmax"]
            )

            if not in_zone:
                continue  # Игрок вне зоны, пропускаем

            # Специальная обработка для зоны "Сосмарк"
            if zone_name == "Сосмарк":
                if clean_name in allowed_players:
                    logging.info(f"Игрок {name} ({clean_name}) в белом списке, пропуск")
                    continue  # Игрок в белом списке, пропускаем

            # Если оповещение еще не срабатывало
            if not alert["triggered"]:
                alert["triggered"] = True
                logging.warning(f"Срабатывание алерта для {name} в {zone_name}")
                return f"Player {name} in {alert['zone']}"

        # Сбрасываем флаг срабатывания если нарушителей нет
        alert["triggered"] = False
        return None

    def process_alerts(self):
        """
        Обработка очереди оповещений.
        Выводит сообщения в лог и на график.
        """
        while not self.alert_queue.empty():
            alert = self.alert_queue.get()
            logging.warning(f"ALERT: {alert}")
            self.show_alert(alert)

    def show_alert(self, message):
        """
        Визуальное отображение оповещения на графике.
        Сообщение показывается в течение 3 секунд.
        """
        # Создаем временную аннотацию
        self.ax.annotate(
            f"! {message} !",
            xy=(0.5, 1.02),  # Позиция в верхней части графика
            xycoords='axes fraction',
            color='red',
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )
        
        # Принудительная перерисовка и задержка
        plt.draw()
        plt.pause(3)
        
        # Удаление сообщения после показа
        if self.ax.texts:
            self.ax.texts[-1].remove()

    def draw_heatmap(self):
        """
        Отрисовка тепловой карты активности игроков.
        Может использовать как текущие, так и исторические данные.
        """
        try:
            # Проверка наличия данных для отрисовки
            if not self.historical_data and not self.current_data:
                return

            heatmap_config = self.config["heatmap"]

            # Выбор источника данных в зависимости от режима
            if heatmap_config["mode"] == "history":
                data = list(self.historical_data)
            else:
                data = [(p["position"]["x"], p["position"]["z"]) for p in self.current_data]

            if not data:
                return

            # Подготовка координат
            x, z = zip(*data)

            # Создание 2D гистограммы
            self.ax.hist2d(
                x, z,
                bins=heatmap_config["bins"],
                cmap=heatmap_config["cmap"],
                alpha=heatmap_config["alpha"],
                zorder=-1,  # Отрисовывается под другими элементами
                range=[
                    [self.config["world_bounds"]["xmin"], self.config["world_bounds"]["xmax"]],
                    [self.config["world_bounds"]["zmin"], self.config["world_bounds"]["zmax"]]
                ]
            )
        except Exception as e:
            logging.error(f"Heatmap error: {str(e)}")

    def update_plot(self, frame):
        """
        Основная функция обновления графика.
        Вызывается периодически для анимации.
        """
        try:
            # Обработка очереди обновлений GUI
            while not self.gui_update_queue.empty():
                update_func = self.gui_update_queue.get()
                update_func()

            # Очистка предыдущего состояния
            self.ax.clear()

            # Отрисовка тепловой карты если включена
            if self.config["heatmap"]["show"]:
                self.draw_heatmap()

            # Последовательная отрисовка элементов
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
        """
        Отрисовка игроков на карте с учетом фильтров.
        """
        with self.data_lock:
            # Фильтрация игроков по выбранным в чекбоксах
            filtered = [p for p in self.current_data 
                       if not self.selected_players or p["name"] in self.selected_players]

        if filtered:
            # Подготовка координат
            x = [p["position"]["x"] for p in filtered]
            z = [p["position"]["z"] for p in filtered]

            # Отрисовка точек игроков
            self.ax.scatter(
                x, z,
                s=self.config["display"]["point_size"],
                c=self.config["display"]["point_color"],
                alpha=self.config["display"]["point_alpha"],
                edgecolors='none',
                zorder=10  # Отрисовывается поверх тепловой карты
            )

            # Добавление подписей к игрокам
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
                    zorder=11  # Отрисовывается поверх точек
                )
                self.label_objects.append(text)

    def draw_zones(self):
        """
        Отрисовка контролируемых зон на карте.
        """
        for zone in self.config["alerts"]["zones"]:
            # Получение границ зоны
            xmin, xmax = zone["bounds"]["xmin"], zone["bounds"]["xmax"]
            zmin, zmax = zone["bounds"]["zmin"], zone["bounds"]["zmax"]
            
            # Отрисовка прямоугольника зоны
            self.ax.add_patch(plt.Rectangle(
                (xmin, zmin), xmax - xmin, zmax - zmin,
                fill=False, 
                edgecolor='red', 
                linestyle='--', 
                linewidth=1,
                zorder=10  # Отрисовывается поверх тепловой карты
            ))
            
            # Добавление названия зоны
            self.ax.text(
                xmin + 50,  # Смещение от угла
                zmin + 50,
                zone["name"],
                color='red', 
                zorder=11  # Отрисовывается поверх других элементов
            )

    def setup_labels(self):
        """
        Настройка осей и заголовка графика.
        """
        # Установка границ отображения
        self.ax.set_xlim(self.config["world_bounds"]["xmin"], self.config["world_bounds"]["xmax"])
        self.ax.set_ylim(self.config["world_bounds"]["zmin"], self.config["world_bounds"]["zmax"])
        
        # Обновление заголовка с текущим временем
        self.ax.set_title(
            f"Player Activity Map ({datetime.now().strftime('%H:%M:%S')})", 
            color='white'
        )
        
        # Настройка сетки
        self.ax.grid(color='#30363d', linestyle='--')

    def update_player_list_text(self):
        """
        Обновление текста в боковой панели со списком игроков.
        """
        with self.data_lock:
            # Сортировка игроков по имени
            players = sorted(self.current_data, key=lambda x: x['name'])
            text_lines = ["Online Players (Overworld):\n"]
            
            # Формирование строк с информацией о позициях
            for player in players:
                x = int(player['position']['x'])
                z = int(player['position']['z'])
                text_lines.append(f"{player['name']: <20} X: {x: >6} Z: {z: >6}")

            # Обновление текстового элемента
            self.player_list_text.set_text("\n".join(text_lines))

    def run(self):
        """Основной цикл приложения."""
        try:
            ani = FuncAnimation(
                self.fig, self.update_plot,
                interval=2000,  # Интервал обновления GUI
                cache_frame_data=False
            )
            plt.show()
        finally:
            self.shutdown()

    def shutdown(self):
        """Корректное завершение работы."""
        self.stop_event.set()
        self.save_history()
        plt.close('all')

if __name__ == "__main__":
    monitor = NoSos()
    monitor.run()
