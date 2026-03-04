from __future__ import annotations
from pathlib import Path

# Project root = parent of /etl
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "raw_layer"

DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
EXPORTS_DIR = DASHBOARD_DIR / "exports"
FINAL_STORY_DIR = PROJECT_ROOT / "final_story"


def ensure_dirs() -> None:
    for d in [DATA_DIR, ANALYSIS_DIR, DASHBOARD_DIR, EXPORTS_DIR, FINAL_STORY_DIR]:
        d.mkdir(parents=True, exist_ok=True)