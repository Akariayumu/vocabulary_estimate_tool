"""词汇测验选择题使用的中文释义。

导入时从 ``data/translations.json`` 加载。
"""

from __future__ import annotations

import json
from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "translations.json"

with open(_DATA_PATH, encoding="utf-8") as _f:
    TRANSLATIONS: dict[str, str] = json.load(_f)
