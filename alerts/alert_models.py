from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

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
    metadata: Optional[Dict[str, Any]] = None
    cooldown: float = 60
