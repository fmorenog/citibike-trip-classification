"""
utils.py
Shared helper functions used across the project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root paths ──────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PRO = ROOT / "data" / "processed"
DB_PATH  = ROOT / "data" / "citibike.duckdb"

# Create directories if they don't exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PRO.mkdir(parents=True, exist_ok=True)


def get_api_key(name: str) -> str:
    """
    Retrieve an API key from environment variables.
    Raises a clear error if the key is missing.
    """
    key = os.getenv(name)
    if not key:
        raise EnvironmentError(
            f"Missing environment variable: {name}\n"
            f"Copy .env.example to .env and add your key."
        )
    return key


def log(msg: str) -> None:
    """Simple timestamped print for pipeline steps."""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
