"""
database.py
Phase 3: DuckDB schema creation and data ingestion.
"""

import duckdb
from src.utils import DATA_RAW, DB_PATH, log


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create the four core tables in DuckDB.
    Safe to re-run — uses CREATE TABLE IF NOT EXISTS.
    """
    log("[3.1] Creating DuckDB schema...")

    con.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            station_id   VARCHAR PRIMARY KEY,
            name         VARCHAR,
            lat          DOUBLE,
            lon          DOUBLE,
            capacity     INTEGER,
            region_id    VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            ride_id              VARCHAR,
            rideable_type        VARCHAR,
            started_at           TIMESTAMPTZ,
            ended_at             TIMESTAMPTZ,
            start_station_name   VARCHAR,
            start_station_id     VARCHAR,
            end_station_name     VARCHAR,
            end_station_id       VARCHAR,
            start_lat            DOUBLE,
            start_lng            DOUBLE,
            end_lat              DOUBLE,
            end_lng              DOUBLE,
            member_casual        VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS weather_hourly (
            datetime_utc     TIMESTAMPTZ PRIMARY KEY,
            temp_c           DOUBLE,
            feels_like_c     DOUBLE,
            humidity_pct     DOUBLE,
            wind_speed_ms    DOUBLE,
            precip_mm        DOUBLE,
            condition_code   INTEGER,
            condition_desc   VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS pluto (
            BBL          VARCHAR,
            landuse      VARCHAR,
            bldgclass    VARCHAR,
            zonedist1    VARCHAR,
            xcoord       INTEGER,
            ycoord       INTEGER,
            latitude     DOUBLE,
            longitude    DOUBLE,
            bct2020      VARCHAR,
            borough      VARCHAR
        )
    """)

    log("[3.1] Schema created — 4 tables ready")


def ingest_stations(con: duckdb.DuckDBPyConnection) -> int:
    """Load stations.parquet into the stations table."""
    log("[3.2] Ingesting stations...")
    con.execute("DELETE FROM stations")
    con.execute(f"""
        INSERT INTO stations
        SELECT * FROM read_parquet('{DATA_RAW}/stations.parquet')
    """)
    count = con.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
    log(f"[3.2] stations: {count:,} rows")
    return count


def ingest_trips(con: duckdb.DuckDBPyConnection) -> int:
    """Load all monthly trip Parquet files into the trips table."""
    log("[3.2] Ingesting trips (this may take a minute)...")
    con.execute("DELETE FROM trips")
    con.execute(f"""
        INSERT INTO trips
        SELECT * FROM read_parquet('{DATA_RAW}/*-citibike-tripdata.parquet')
    """)
    count = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    log(f"[3.2] trips: {count:,} rows")
    return count


def ingest_pluto(con: duckdb.DuckDBPyConnection) -> int:
    """Load pluto.parquet into the pluto table."""
    log("[3.2] Ingesting PLUTO...")
    con.execute("DROP TABLE IF EXISTS pluto")
    con.execute(f"""
        CREATE TABLE pluto AS
        SELECT * FROM read_parquet('{DATA_RAW}/pluto.parquet')
    """)
    count = con.execute("SELECT COUNT(*) FROM pluto").fetchone()[0]
    log(f"[3.2] pluto: {count:,} rows")
    return count


def validate(con: duckdb.DuckDBPyConnection) -> None:
    """
    Step 3.3 — Basic validation checks:
    - Row counts per table
    - Trip station ID match rate against stations table
    - Date range of trips
    """
    log("[3.3] Validating...")

    for table in ["stations", "trips", "pluto"]:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        log(f"    {table}: {n:,} rows")

    matched = con.execute("""
        SELECT COUNT(*) FROM trips t
        INNER JOIN stations s ON t.start_station_name = s.name
    """).fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    pct = matched / total * 100
    log(f"    Station ID match rate: {pct:.1f}% ({matched:,} / {total:,})")

    date_range = con.execute("""
        SELECT MIN(started_at), MAX(started_at) FROM trips
    """).fetchone()
    log(f"    Trip date range: {date_range[0]} -> {date_range[1]}")

    log("[3.3] Validation complete")
