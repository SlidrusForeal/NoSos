# NoSos — Мониторинг игроков Minecraft сервера

*Мониторинг игроков в реальном времени с тепловой картой и зонами оповещений.*

## Описание
NoSos — это инструмент для мониторинга активности игроков на Minecraft сервере. Он предоставляет:
- Реальное отслеживание позиций игроков.
- Визуализацию тепловой карты активности.
- Оповещения о вторжениях в защищенные зоны.
- Аналитику аномалий (высокая скорость, телепортация).
- Интеграцию с Telegram для уведомлений и управления.

## Особенности
- **Карта активности**: Отображение игроков, зон и тепловой карты.
- **Настраиваемые зоны**: Оповещения при входе в запрещенные зоны.
- **Telegram бот**: Управление доступом, отправка уведомлений, команды для админов.
- **Аналитика**: Отчеты по времени игроков, топ активных зон.
- **Безопасность**: Логирование событий, контроль доступа через Telegram.

## Установка
1. **Зависимости**:  
   Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
   (Файл `requirements.txt` должен включать: `matplotlib`, `requests`, `python-telegram-bot`, `sqlite3`, `numpy`, `pandas`, `pyyaml`)

2. **Настройка**:  
   Отредактируйте `config.yaml`:
   - `players_url`: URL для получения данных игроков.
   - `telegram.token` и `telegram.chat_id`: Токен бота и ID чата.
   - Настройте зоны в `alerts.zones`.

3. **Запуск**:  
   Выполните:
   ```bash
   python main.py
   ```

## Конфигурация
Ключевые параметры `config.yaml`:
```yaml
players_url: "URL_игроков.json"
world_bounds: # Границы мира
  xmin: -10145
  xmax: 10145
  zmin: -10078
  zmax: 10078
telegram:
  token: "ТОКЕН_БОТА"
  chat_id: "ID_ЧАТА"
alerts:
  zones: # Список зон с настройками оповещений
    - name: "Спавн"
      bounds: {xmin: -500, xmax: 500, zmin: -500, zmax: 500}
      alert_level: "INFO"
```

## Использование
### Telegram команды
- Для пользователей:
  - `/start` — регистрация.
  - `/subscribe` — подписка на уведомления.
  - `/history` — топ активных игроков.
- Для админов:
  - `/users` — список пользователей.
  - `/approve <ID>` — одобрить доступ.
  - `/heatmap` — отчет по активным зонам.

### Интерфейс
- График отображает позиции игроков, зоны и тепловую карту.
- Оповещения выводятся в верхней части экрана.

## Примеры запросов
1. Проверка аномалий скорости:
   ```
   /anomalies 60 120
   ```
2. Отчет по игроку:
   ```
   /player_report Игрок123
   ```

## Лицензия
Проект распространяется под лицензией MIT. Подробнее см. в файле `LICENSE`.
