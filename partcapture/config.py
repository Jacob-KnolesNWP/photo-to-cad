"""Runtime paths and limits."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"

DATA_DIR = Path(os.environ.get("PARTCAPTURE_DATA", "data")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "jobs.sqlite3"

MAX_FLAT_PHOTOS = 12
MAX_OBJECT_PHOTOS = 60
MIN_OBJECT_PHOTOS = 8
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
FLAT_MAX_EDGE = 2800
OBJECT_MAX_EDGE = 1400


def ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
