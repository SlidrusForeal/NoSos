import time
import re
import unicodedata
import numpy as np
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from telegram.helpers import escape_markdown
from alerts.alert_models import Alert, AlertLevel

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
        self.max_speed = config.get("max_speed", 50)
        self.teleport_threshold = config.get("teleport_threshold", 100)
        self.player_positions: Dict[str, Tuple[Optional[Tuple[float, float]], float]] = {}

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
                distance = self._calculate_distance(last_pos, current_pos)
                time_diff = current_time - last_time
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
                print(f"Ошибка обработки игрока {player.get('name')}: {e}")
        return alerts

    def _get_player_history(self, player_id: str) -> Tuple[Optional[Tuple[float, float]], float]:
        return self.player_positions.get(player_id, (None, 0.0))

    def _update_player_history(self, player_id: str, pos: Tuple[float, float], timestamp: float):
        self.player_positions[player_id] = (pos, timestamp)

    @staticmethod
    def _calculate_distance(pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _create_speed_alert(self, player: Dict, speed: float) -> Alert:
        safe_player = escape_markdown(player['name'], version=2)
        safe_speed = escape_markdown(f"{speed:.1f}", version=2)
        return Alert(
            message=f"Игрок {safe_player} движется со скоростью {safe_speed} блоков/сек",
            level=self.alert_level,
            source="movement_anomaly",
            timestamp=datetime.now(),
            metadata={
                "player": player['name'],
                "speed": speed,
                "position": player["position"]
            },
            cooldown=self.cooldown
        )

    def _create_teleport_alert(self, player: Dict, distance: float) -> Alert:
        safe_player = escape_markdown(player['name'], version=2)
        safe_distance = escape_markdown(f"{distance:.1f}", version=2)
        return Alert(
            message=f"Игрок {safe_player} переместился на {safe_distance} блоков мгновенно",
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
        intruders = []
        for player in data.get("players", []):
            pos = player.get("position", {})
            x, z = pos.get("x", 0), pos.get("z", 0)
            if not self._in_zone(x, z):
                continue
            norm_name = self._normalize_name(player.get("name", ""))
            if norm_name in allowed:
                continue
            # Сохраняем дополнительные данные: armor и health (если есть)
            intruders.append({
                "name": player.get("name", "Unknown"),
                "armor": player.get("armor", "N/A"),
                "health": player.get("health", "N/A")
            })
        if intruders:
            alert_id = f"{zone_name}_intrusion"
            if self._should_trigger(alert_id):
                alerts.append(self._create_alert(zone_name, intruders))
                self._update_cooldown(alert_id)
        return alerts

    def _create_alert(self, zone_name: str, intruders: List[dict]) -> Alert:
        # Формируем сообщение с данными об armor и health для каждого игрока
        intruders_info = ", ".join(
            [f"{p['name']} (Armor: {p['armor']}, Health: {p['health']})" for p in intruders]
        )
        return Alert(
            message=f"Обнаружены игроки в зоне {zone_name}: {intruders_info}",
            level=self.alert_level,
            source="zone_intrusion",
            timestamp=datetime.now(),
            metadata={
                "zone": zone_name,
                "players": intruders,
                "count": len(intruders)
            },
            cooldown=self.cooldown
        )

class PlayerCountRule(BaseAlertRule):
    def check_conditions(self, data: Dict) -> List[Alert]:
        current = len(data.get("players", []))
        max_players = self.config.get("max_players", 50)
        if current > max_players:
            return [Alert(
                message=f"Превышен лимит игроков: {current}/{max_players}",
                level=AlertLevel.CRITICAL,
                source="player_limit",
                timestamp=datetime.now(),
                metadata={"current": current, "max": max_players}
            )]
        return []
