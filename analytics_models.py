"""
Data models for Telegram channel analytics.
Extracts military intelligence: equipment losses, casualties, operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(Enum):
    """Which side suffered the loss."""
    RUSSIAN = "russian"
    UKRAINIAN = "ukrainian"
    UNKNOWN = "unknown"


class OutcomeType(Enum):
    """What happened to the equipment."""
    DESTROYED = "destroyed"
    DAMAGED = "damaged"
    CAPTURED = "captured"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"


class EquipmentCategory(Enum):
    """Category of military equipment."""
    TANK = "tank"
    IFV = "ifv"  # Infantry Fighting Vehicle
    APC = "apc"  # Armored Personnel Carrier
    ARTILLERY = "artillery"
    MLRS = "mlrs"  # Multiple Launch Rocket System
    HELICOPTER = "helicopter"
    AIRCRAFT = "aircraft"
    UAV = "uav"  # Drone
    AIR_DEFENSE = "air_defense"
    LOGISTICS = "logistics"
    NAVAL = "naval"
    OTHER = "other"


@dataclass
class EquipmentLoss:
    """Single equipment loss event."""
    equipment_type: str  # Specific type, e.g., "T-72B3", "Ka-52"
    category: EquipmentCategory
    side: Side
    outcome: OutcomeType
    count: int = 1
    confidence: float = 0.5
    source_msg_id: int = 0
    timestamp: Optional[datetime] = None
    location: str = ""


@dataclass
class CasualtyReport:
    """Casualty information from a post."""
    side: Side
    killed: int = 0
    wounded: int = 0
    captured: int = 0
    confidence: float = 0.5
    source_msg_id: int = 0
    timestamp: Optional[datetime] = None
    unit: str = ""


@dataclass
class AnalyticsResult:
    """Complete analytics result for a single message."""
    msg_id: int
    equipment_losses: list[EquipmentLoss] = field(default_factory=list)
    casualties: list[CasualtyReport] = field(default_factory=list)
    location: str = ""
    is_victory_claim: bool = False
    is_loss_admission: bool = False
    raw_ai_response: str = ""
    processed_at: Optional[datetime] = None


@dataclass
class AggregatedStats:
    """Aggregated statistics over analyzed messages."""
    total_messages: int = 0
    analyzed_messages: int = 0

    # Equipment losses by category and side
    equipment_by_category: dict[EquipmentCategory, dict[Side, int]] = field(default_factory=dict)
    equipment_by_type: dict[str, dict[Side, int]] = field(default_factory=dict)

    # Casualty totals by side
    casualties_by_side: dict[Side, dict[str, int]] = field(default_factory=dict)

    # Victory/loss claims
    victory_claims: int = 0
    loss_admissions: int = 0


# Equipment type mappings for common military hardware
EQUIPMENT_MAPPINGS: dict[str, tuple[EquipmentCategory, str]] = {
    # Tanks
    "t-72": (EquipmentCategory.TANK, "T-72"),
    "t-80": (EquipmentCategory.TANK, "T-80"),
    "t-90": (EquipmentCategory.TANK, "T-90"),
    "t-62": (EquipmentCategory.TANK, "T-62"),
    "t-55": (EquipmentCategory.TANK, "T-55"),
    "leopard": (EquipmentCategory.TANK, "Leopard 2"),
    "abrams": (EquipmentCategory.TANK, "M1 Abrams"),
    "challenger": (EquipmentCategory.TANK, "Challenger 2"),
    "pt-91": (EquipmentCategory.TANK, "PT-91"),

    # Helicopters
    "ka-52": (EquipmentCategory.HELICOPTER, "Ka-52 Alligator"),
    "ka-50": (EquipmentCategory.HELICOPTER, "Ka-50"),
    "mi-28": (EquipmentCategory.HELICOPTER, "Mi-28 Havoc"),
    "mi-24": (EquipmentCategory.HELICOPTER, "Mi-24 Hind"),
    "mi-8": (EquipmentCategory.HELICOPTER, "Mi-8"),
    "mi-17": (EquipmentCategory.HELICOPTER, "Mi-17"),
    "apache": (EquipmentCategory.HELICOPTER, "AH-64 Apache"),

    # Aircraft
    "su-25": (EquipmentCategory.AIRCRAFT, "Su-25 Frogfoot"),
    "su-34": (EquipmentCategory.AIRCRAFT, "Su-34 Fullback"),
    "su-35": (EquipmentCategory.AIRCRAFT, "Su-35 Flanker-E"),
    "su-30": (EquipmentCategory.AIRCRAFT, "Su-30"),
    "su-27": (EquipmentCategory.AIRCRAFT, "Su-27"),
    "mig-29": (EquipmentCategory.AIRCRAFT, "MiG-29"),
    "mig-31": (EquipmentCategory.AIRCRAFT, "MiG-31"),
    "f-16": (EquipmentCategory.AIRCRAFT, "F-16"),
    "a-50": (EquipmentCategory.AIRCRAFT, "A-50 AWACS"),
    "tu-22": (EquipmentCategory.AIRCRAFT, "Tu-22M"),
    "tu-95": (EquipmentCategory.AIRCRAFT, "Tu-95"),

    # IFVs
    "bmp-1": (EquipmentCategory.IFV, "BMP-1"),
    "bmp-2": (EquipmentCategory.IFV, "BMP-2"),
    "bmp-3": (EquipmentCategory.IFV, "BMP-3"),
    "bmd": (EquipmentCategory.IFV, "BMD"),
    "bradley": (EquipmentCategory.IFV, "M2 Bradley"),
    "cv90": (EquipmentCategory.IFV, "CV90"),
    "marder": (EquipmentCategory.IFV, "Marder"),

    # APCs
    "btr-80": (EquipmentCategory.APC, "BTR-80"),
    "btr-82": (EquipmentCategory.APC, "BTR-82A"),
    "btr-70": (EquipmentCategory.APC, "BTR-70"),
    "mt-lb": (EquipmentCategory.APC, "MT-LB"),
    "m113": (EquipmentCategory.APC, "M113"),
    "stryker": (EquipmentCategory.APC, "Stryker"),

    # Artillery
    "howitzer": (EquipmentCategory.ARTILLERY, "Howitzer"),
    "2s19": (EquipmentCategory.ARTILLERY, "2S19 Msta"),
    "2s1": (EquipmentCategory.ARTILLERY, "2S1 Gvozdika"),
    "2s3": (EquipmentCategory.ARTILLERY, "2S3 Akatsiya"),
    "d-30": (EquipmentCategory.ARTILLERY, "D-30"),
    "m777": (EquipmentCategory.ARTILLERY, "M777"),
    "pzh 2000": (EquipmentCategory.ARTILLERY, "PzH 2000"),
    "caesar": (EquipmentCategory.ARTILLERY, "CAESAR"),
    "krab": (EquipmentCategory.ARTILLERY, "Krab"),

    # MLRS
    "himars": (EquipmentCategory.MLRS, "HIMARS"),
    "grad": (EquipmentCategory.MLRS, "BM-21 Grad"),
    "uragan": (EquipmentCategory.MLRS, "BM-27 Uragan"),
    "smerch": (EquipmentCategory.MLRS, "BM-30 Smerch"),
    "tornado": (EquipmentCategory.MLRS, "Tornado-G"),
    "m270": (EquipmentCategory.MLRS, "M270 MLRS"),

    # Drones/UAVs
    "lancet": (EquipmentCategory.UAV, "Lancet"),
    "shahed": (EquipmentCategory.UAV, "Shahed-136"),
    "geran": (EquipmentCategory.UAV, "Geran-2"),
    "bayraktar": (EquipmentCategory.UAV, "Bayraktar TB2"),
    "orlan": (EquipmentCategory.UAV, "Orlan-10"),
    "zala": (EquipmentCategory.UAV, "ZALA"),
    "mavic": (EquipmentCategory.UAV, "DJI Mavic"),
    "fpv": (EquipmentCategory.UAV, "FPV Drone"),

    # Air Defense
    "s-300": (EquipmentCategory.AIR_DEFENSE, "S-300"),
    "s-400": (EquipmentCategory.AIR_DEFENSE, "S-400"),
    "buk": (EquipmentCategory.AIR_DEFENSE, "Buk"),
    "tor": (EquipmentCategory.AIR_DEFENSE, "Tor"),
    "pantsir": (EquipmentCategory.AIR_DEFENSE, "Pantsir"),
    "tunguska": (EquipmentCategory.AIR_DEFENSE, "Tunguska"),
    "patriot": (EquipmentCategory.AIR_DEFENSE, "Patriot"),
    "nasams": (EquipmentCategory.AIR_DEFENSE, "NASAMS"),
    "iris-t": (EquipmentCategory.AIR_DEFENSE, "IRIS-T"),

    # Logistics/Support
    "kamaz": (EquipmentCategory.LOGISTICS, "KamAZ Truck"),
    "ural": (EquipmentCategory.LOGISTICS, "Ural Truck"),
    "fuel truck": (EquipmentCategory.LOGISTICS, "Fuel Truck"),
    "ammunition": (EquipmentCategory.LOGISTICS, "Ammo Truck/Depot"),

    # Naval
    "moskva": (EquipmentCategory.NAVAL, "Moskva Cruiser"),
    "frigate": (EquipmentCategory.NAVAL, "Frigate"),
    "patrol boat": (EquipmentCategory.NAVAL, "Patrol Boat"),
    "landing ship": (EquipmentCategory.NAVAL, "Landing Ship"),
}

# Side indicators - words that suggest which side
SIDE_INDICATORS: dict[Side, list[str]] = {
    Side.RUSSIAN: [
        "russian", "russia", "rf", "enemy", "occupier", "orc",
        "ruzzian", "invader", "aggressor", "rashist"
    ],
    Side.UKRAINIAN: [
        "ukrainian", "ukraine", "ua", "afu", "zsu", "defender",
        "our", "ours", "friendly", "allied"
    ],
}
