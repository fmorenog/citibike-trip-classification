"""
data_download.py
Phase 2: Programmatic data collection for all four sources.

Usage (from project root):
    python src/data_download.py

Or import individual functions into the notebook:
    from src.data_download import download_citibike_trips, fetch_gbfs_stations, ...
"""

import io
import json
import time
import zipfile
import calendar
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.utils import DATA_RAW, DATA_PRO, get_api_key, log


# ── Constants ────────────────────────────────────────────────────────────────

CITIBIKE_S3_BASE    = "https://s3.amazonaws.com/tripdata"
GBFS_STATION_INFO_URL = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json"
PLUTO_URL = (
    "https://data.cityofnewyork.us/api/views/64uk-42ks/rows.csv?accessType=DOWNLOAD"
)

# Only keep columns we need — discard everything else at read time
TRIP_COLUMNS = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id",
    "end_station_name",   "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng",
    "member_casual",
]


# ── 2.1  Citi Bike trip data ──────────────────────────────────────────────────

def download_citibike_trips(
    year: int = 2024,
    months=None,
    out_dir: Path = DATA_RAW,
    overwrite: bool = False,
) -> list:
    """
    Download monthly Citi Bike trip ZIP files from the public S3 bucket,
    unzip in memory, select needed columns, and save as Parquet.

    Parameters
    ----------
    year     : calendar year (default 2024)
    months   : list of month ints 1-12; None = all 12
    out_dir  : destination directory  (data/raw/)
    overwrite: if False, skip files that already exist on disk

    Returns
    -------
    List of Parquet paths written.
    """
    if months is None:
        months = list(range(1, 13))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for month in months:
        yyyymm = f"{year}{month:02d}"
        parquet_path = out_dir / f"{yyyymm}-citibike-tripdata.parquet"

        if parquet_path.exists() and not overwrite:
            log(f"[2.1] {yyyymm} already on disk — skipping")
            written.append(parquet_path)
            continue

        # Citi Bike has used two naming styles over the years
        candidates = [
            f"{CITIBIKE_S3_BASE}/{yyyymm}-citibike-tripdata.csv.zip",
            f"{CITIBIKE_S3_BASE}/{yyyymm}-citibike-tripdata.zip",
        ]

        url = _find_url(candidates)
        if url is None:
            log(f"[2.1] WARNING: no file found for {yyyymm} — skipping")
            continue

        log(f"[2.1] Downloading {url}")
        try:
            response = _get_with_retry(url)
        except requests.RequestException as e:
            log(f"[2.1] ERROR downloading {yyyymm}: {e}")
            continue

        # Unzip in memory — some months ship multiple CSVs in one ZIP
        dfs = []
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            for csv_name in csv_names:
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, usecols=lambda c: c in TRIP_COLUMNS)
                    dfs.append(df)

        if not dfs:
            log(f"[2.1] WARNING: no CSVs inside zip for {yyyymm}")
            continue

        combined = pd.concat(dfs, ignore_index=True)
        combined = _standardise_trip_dtypes(combined)
        combined.to_parquet(parquet_path, index=False)
        log(f"[2.1] {yyyymm}: {len(combined):,} rows → {parquet_path.name}")
        written.append(parquet_path)

    log(f"[2.1] Done — {len(written)} Parquet files in {out_dir}")
    return written


def _find_url(candidates):
    """Return the first URL that responds HTTP 200, or None."""
    for url in candidates:
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


def _get_with_retry(url, retries=3, backoff=5.0):
    """GET with simple exponential-ish retry."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries:
                raise
            log(f"    Attempt {attempt} failed ({e}), retrying in {backoff}s...")
            time.sleep(backoff)


def _standardise_trip_dtypes(df):
    """Parse timestamps and enforce correct dtypes on a raw trip DataFrame."""
    for col in ("started_at", "ended_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    str_cols = [
        "ride_id", "rideable_type",
        "start_station_name", "start_station_id",
        "end_station_name",   "end_station_id",
        "member_casual",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ("start_lat", "start_lng", "end_lat", "end_lng"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── 2.2  GBFS station metadata ────────────────────────────────────────────────

def fetch_gbfs_stations(
    url: str = GBFS_STATION_INFO_URL,
    out_dir: Path = DATA_RAW,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Fetch station metadata from the Citi Bike GBFS station_information endpoint.
    Saves to data/raw/stations.parquet.

    Fields: station_id, name, lat, lon, capacity, region_id
    """
    out_path = out_dir / "stations.parquet"
    if out_path.exists() and not overwrite:
        log("[2.2] stations.parquet already on disk — loading")
        return pd.read_parquet(out_path)

    log(f"[2.2] Fetching GBFS station info from:\n      {url}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"[2.2] GBFS request failed: {e}") from e

    payload = r.json()
    stations_raw = payload["data"]["stations"]

    keep = ["station_id", "name", "lat", "lon", "capacity", "region_id"]
    df = pd.DataFrame(stations_raw)
    df = df[[c for c in keep if c in df.columns]].copy()

    df["station_id"] = df["station_id"].astype(str).str.strip()
    for col in ("capacity", "lat", "lon"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log(f"[2.2] Saved {len(df):,} stations → {out_path.name}")
    return df


# ── 2.3  OpenWeatherMap historical weather ────────────────────────────────────

def fetch_weather_history(
    year: int = 2024,
    months=None,
    lat: float = 40.7789,   # Central Park, NYC
    lon: float = -73.9692,
    out_dir: Path = DATA_RAW,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather from OpenWeatherMap One Call API 3.0.
    Requires OPENWEATHER_API_KEY set in your .env file.

    Makes one API call per day (free tier: 1,000 calls/day).
    A full year = 366 calls, taking ~7 minutes with the 1-second rate limit.

    Saves to data/raw/weather_hourly.parquet.

    Fields: datetime_utc, temp_c, feels_like_c, humidity_pct,
            wind_speed_ms, precip_mm, condition_code, condition_desc
    """
    out_path = out_dir / "weather_hourly.parquet"
    if out_path.exists() and not overwrite:
        log("[2.3] weather_hourly.parquet already on disk — loading")
        return pd.read_parquet(out_path)

    api_key = get_api_key("OPENWEATHER_API_KEY")

    if months is None:
        months = list(range(1, 13))

    # One timestamp per day (midnight UTC) → one API call per day
    timestamps = []
    for month in months:
        _, days_in_month = calendar.monthrange(year, month)
        for day in range(1, days_in_month + 1):
            dt = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
            timestamps.append(int(dt.timestamp()))

    log(f"[2.3] Fetching {len(timestamps)} days of weather (this takes ~{len(timestamps)//55} min)")
    base_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

    all_records = []
    for ts in tqdm(timestamps, desc="[2.3] Weather API"):
        params = {
            "lat": lat, "lon": lon, "dt": ts,
            "appid": api_key, "units": "metric",
        }
        try:
            r = requests.get(base_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            log(f"[2.3] WARNING: failed for ts={ts}: {e}")
            continue

        for hour in data.get("data", []):
            record = {
                "datetime_utc":  pd.Timestamp(hour["dt"], unit="s", tz="UTC"),
                "temp_c":        hour.get("temp"),
                "feels_like_c":  hour.get("feels_like"),
                "humidity_pct":  hour.get("humidity"),
                "wind_speed_ms": hour.get("wind_speed"),
                "precip_mm":     (hour.get("rain", {}).get("1h", 0.0)
                                  + hour.get("snow", {}).get("1h", 0.0)),
                "condition_code": (hour["weather"][0]["id"]
                                   if hour.get("weather") else None),
                "condition_desc": (hour["weather"][0]["description"]
                                   if hour.get("weather") else None),
            }
            all_records.append(record)

        time.sleep(1.1)   # Stay within 1 call/sec rate limit

    df = pd.DataFrame(all_records).sort_values("datetime_utc").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log(f"[2.3] Saved {len(df):,} hourly records → {out_path.name}")
    return df


# ── 2.4  NYC PLUTO land use data ──────────────────────────────────────────────

def download_pluto(
    url: str = PLUTO_URL,
    out_dir: Path = DATA_RAW,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download NYC MapPLUTO land use data from NYC Open Data.
    Keeps only the columns needed for station land use tagging.

    Saves to data/raw/pluto.parquet.

    Key fields: BBL, LandUse, BldgClass, ZoneDist1,
                Latitude, Longitude, Borough, BoroCT2020
    """
    out_path = out_dir / "pluto.parquet"
    if out_path.exists() and not overwrite:
        log("[2.4] pluto.parquet already on disk — loading")
        return pd.read_parquet(out_path)

    log("[2.4] Downloading NYC PLUTO from NYC Open Data (~300 MB)...")

    keep_cols = [
    "BBL", "landuse", "bldgclass", "zonedist1",
    "xcoord", "ycoord", "latitude", "longitude",
    "bct2020", "borough",
    ]

    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        df = pd.read_csv(
            io.StringIO(r.content.decode("utf-8")),
            usecols=lambda c: c in keep_cols,
            dtype=str,
            low_memory=False,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"[2.4] Failed to download PLUTO: {e}") from e

    for col in ("Latitude", "Longitude", "XCoord", "YCoord"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log(f"[2.4] Saved {len(df):,} PLUTO lots → {out_path.name}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download all Citi Bike project data")
    parser.add_argument("--year",         type=int, default=2024)
    parser.add_argument("--months",       type=int, nargs="+", default=None)
    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument("--skip-weather", action="store_true",
                        help="Skip OpenWeatherMap calls (useful if no API key yet)")
    args = parser.parse_args()

    log("=== Phase 2: Data Collection ===")

    log("--- Step 2.1: Citi Bike trip data ---")
    download_citibike_trips(year=args.year, months=args.months, overwrite=args.overwrite)

    log("--- Step 2.2: GBFS station metadata ---")
    fetch_gbfs_stations(overwrite=args.overwrite)

    if not args.skip_weather:
        log("--- Step 2.3: OpenWeatherMap weather ---")
        fetch_weather_history(year=args.year, months=args.months, overwrite=args.overwrite)
    else:
        log("--- Step 2.3: Skipped (--skip-weather) ---")

    log("--- Step 2.4: NYC PLUTO land use ---")
    download_pluto(overwrite=args.overwrite)

    log("=== Phase 2 complete ===")
