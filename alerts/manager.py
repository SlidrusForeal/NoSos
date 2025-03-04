import time
from collections import deque
from functools import lru_cache
import logging
from datetime import datetime
from alerts.alert_models import Alert
from alerts.rules import BaseAlertRule

class AlertManager:
    def __init__(self, monitor):
        self.rules = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.monitor = monitor

    @lru_cache(maxsize=100)
    def _generate_alert_id(self, source: str, message: str) -> str:
        return f"{source}_{hash(message)}"

    def load_rules_from_config(self, config: dict):
        rule_registry = {
            "zone_intrusion": lambda conf: __import__("alerts.rules", fromlist=["ZoneIntrusionRule"]).ZoneIntrusionRule(conf),
            "player_limit": lambda conf: __import__("alerts.rules", fromlist=["PlayerCountRule"]).PlayerCountRule(conf),
            "movement_anomaly": lambda conf: __import__("alerts.rules", fromlist=["MovementAnomalyRule"]).MovementAnomalyRule(conf)
        }
        for rule_config in config.get("zones", []):
            self.rules.append(rule_registry["zone_intrusion"](rule_config))
        if "limits" in config:
            self.rules.append(rule_registry["player_limit"](config["limits"]))
        if "movement_anomaly" in config:
            self.rules.append(rule_registry["movement_anomaly"](config["movement_anomaly"]))

    def process_data(self, data: dict):
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
            self.monitor.send_notifications(alert)
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
        self._clean_expired_alerts()

    def _clean_expired_alerts(self):
        current_time = time.time()
        to_remove = []
        for alert_id, alert in list(self.active_alerts.items()):
            expire_time = alert.timestamp.timestamp() + alert.cooldown
            if current_time > expire_time:
                to_remove.append(alert_id)
        for aid in to_remove:
            if aid in self.active_alerts:
                del self.active_alerts[aid]

    def get_alert_history(self, limit=50):
        return list(self.alert_history)[-limit:]
