import re
import logging
import asyncio
from datetime import datetime
from telegram.helpers import escape_markdown
from parser.player_parser import PlayerParser
from utils.helpers import clean_html_tags

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
        zone_activity = {}
        for zone, players in self.monitor.zone_time.items():
            if isinstance(players, dict):
                zone_activity[zone] = sum(players.values())
            else:
                zone_activity[zone] = players
        sorted_zones = sorted(zone_activity.items(), key=lambda x: x[1], reverse=True)[:5]
        report_lines = [f"• {zone}: {int(time // 60)} минут" for zone, time in sorted_zones]
        return "🔥 Топ-5 активных зон:\n" + "\n".join(report_lines)

    async def generate_player_report(self, player_name: str) -> str:
        logging.debug(f"Запрос данных для игрока: {player_name}")
        try:
            # Очищаем кэш перед новым запросом
            PlayerParser.clear_cache()
            player_history = self.monitor.get_player_history(player_name, limit=5)
            last_position = self.monitor.get_last_position(player_name)
            player_data = await PlayerParser.parse_player_profile(player_name)
        except Exception as e:
            logging.error(f"Ошибка получения данных: {str(e)}")
            return f"❌ Ошибка при получении данных игрока {escape_markdown(player_name, version=2)}"

        report = [
            f"📊 *Отчёт по игроку* `{escape_markdown(player_name, version=2)}`",
            f"🕒 Последняя активность: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        ]

        # Добавляем информацию из текущих данных об armor и health, если есть
        for p in self.monitor.current_data:
            if p["name"].lower() == player_name.lower():
                armor = p.get("armor", "N/A")
                health = p.get("health", "N/A")
                report.append(f"🛡 Броня: {armor} | ❤️ Здоровье: {health}")
                break

        if last_position:
            x, z, _ = last_position
            zone_name = self.monitor.get_zone_name_by_coordinates(x, z)
            if zone_name:
                report.append(f"📍 Последняя зона: {escape_markdown(zone_name, version=2)}")
            else:
                report.append(f"📍 Последние координаты: `X: {int(x)} Z: {int(z)}`")

        if player_history:
            unique_zones = set()
            for x, z, _ in player_history:
                zone_name = self.monitor.get_zone_name_by_coordinates(x, z)
                if zone_name:
                    unique_zones.add(zone_name)
                else:
                    unique_zones.add(f"X: {int(x)}, Z: {int(z)}")
            report.append("\n🔍 *История перемещений:*\n" + "\n".join(f"• {zone}" for zone in unique_zones))

        if player_data:
            sections = [
                ("📱 Соцсети", player_data.get('socials')),
                ("🏅 Роли", player_data.get('roles')),
                ("📈 Статистика", player_data.get('stats')),
                ("🃏 РП-карточки", [
                    f"{card['h3']}: {clean_html_tags(card['p'])}"
                    for card in player_data.get('rp_cards', [])
                ]),
                ("💎 Премиум статус", [clean_html_tags(player_data.get('player_plus'))])
            ]
            for title, data in sections:
                if data and any(data):
                    cleaned_data = []
                    for item in data:
                        if isinstance(item, tuple) and len(item) == 2:
                            name, link = item
                            cleaned_data.append(f"{escape_markdown(name, version=2)}: {link}")
                        else:
                            item = item.replace("sports_esports", "🎮").replace("emoji_events", "🏆")
                            item = re.sub(r'\s+', ' ', item)
                            item = re.sub(r'\s+([.,!?;:])', r'\1', item)
                            cleaned_data.append(item)
                    report.append(f"\n{title}:\n" + "\n".join(f"• {item}" for item in cleaned_data if item))

        if len(report) == 2:
            report.append("\nℹ️ Дополнительные данные не найдены")

        return "\n".join(report)
