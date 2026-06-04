"""
Shared Chicago crime -> ML features for training (train_chicago_safety.py)
and inference (colorpredict3.load_chicago_grid).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List

import pandas as pd

# Types treated as high person-safety risk for training labels (1 = unsafe context)
UNSAFE_PRIMARY_TYPES = frozenset(
    {
        "CRIM SEXUAL ASSAULT",
        "SEX OFFENSE",
        "ASSAULT",
        "BATTERY",
        "ROBBERY",
        "HOMICIDE",
        "CRIMINAL SEXUAL ASSAULT",
        "KIDNAPPING",
        "STALKING",
        "HUMAN TRAFFICKING",
        "WEAPONS VIOLATION",
        "OFFENSE INVOLVING CHILDREN",
        "CRIMINAL DAMAGE",
        "INTIMIDATION",
    }
)

# Clearly lower priority for a *person* risk classifier (label 0)
SAFE_PRIMARY_TYPES = frozenset(
    {
        "THEFT",
        "BURGLARY",
        "MOTOR VEHICLE THEFT",
        "DECEPTIVE PRACTICE",
        "CRIMINAL TRESPASS",
        "NARCOTICS",
        "PUBLIC PEACE VIOLATION",
        "LIQUOR LAW VIOLATION",
        "GAMBLING",
        "PROSTITUTION",
    }
)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y"}


def _parse_hour(date_str: Any) -> int:
    if not date_str:
        return 12
    try:
        return int(datetime.fromisoformat(str(date_str).replace("Z", "")).hour)
    except Exception:
        return 12


def training_label(primary_type: str, domestic: int) -> int:
    pt = (primary_type or "UNKNOWN").strip().upper()
    if pt in UNSAFE_PRIMARY_TYPES:
        return 1
    if pt in SAFE_PRIMARY_TYPES:
        return 0
    if domestic and pt in {"BATTERY", "ASSAULT", "OTHER OFFENSE"}:
        return 1
    return 0


def api_record_to_row(item: dict) -> dict | None:
    try:
        lat = float(item.get("latitude"))
        lon = float(item.get("longitude"))
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    pt = str(item.get("primary_type") or "UNKNOWN").strip()[:120]
    return {
        "latitude": lat,
        "longitude": lon,
        "hour": _parse_hour(item.get("date")),
        "arrest": 1 if _to_bool(item.get("arrest")) else 0,
        "domestic": 1 if _to_bool(item.get("domestic")) else 0,
        "primary_type": pt or "UNKNOWN",
    }


def records_to_row_dicts(records: List[dict]) -> List[dict]:
    out: List[dict] = []
    for item in records:
        row = api_record_to_row(item)
        if row:
            out.append(row)
    return out


def rows_to_feature_frame(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["latitude", "longitude", "hour", "arrest", "domestic", "primary_type"]
        )
    return pd.DataFrame(rows)[
        ["latitude", "longitude", "hour", "arrest", "domestic", "primary_type"]
    ]


def add_training_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["unsafe"] = [
        training_label(r["primary_type"], int(r["domestic"]))
        for r in out.to_dict("records")
    ]
    return out
