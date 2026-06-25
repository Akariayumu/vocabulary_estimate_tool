"""Chinese glosses for vocabulary quiz multiple-choice questions.

Loaded from ``data/translations.json`` at import time.
"""

from __future__ import annotations

import json
from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "translations.json"

with open(_DATA_PATH, encoding="utf-8") as _f:
    TRANSLATIONS: dict[str, str] = json.load(_f)
