"""
pipeline.py
Phase 6: TripClassifierPipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.utils import DATA_PRO, log


class TripClassifierPipeline:
    """
    End-to-end pipeline for classifying Citi Bike trips as
    commuter or recreational.

    Usage
    -----
    pipeline = TripClassifierPipeline()
    pipeline.load()
    pipeline.engineer_features()
    X_train, X_test, y_train, y_test = pipeline.split()
    """

    # Features used for classification
    FEATURE_COLS = [
    'duration_min', 'duration_log',
    'hour', 'day_of_week', 'month',
    'is_weekend', 'is_holiday', 'is_rush_hour',
    'is_member',
    'start_anchored', 'end_anchored',
    'rideable_electric',
     ]

    def __init__(self, data_dir: Path = DATA_PRO):
        self.data_dir = data_dir
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        log("[6] TripClassifierPipeline initialised")

    def load(self):
        """Load the labeled dataset from disk."""
        path = self.data_dir / "trips_labeled.parquet"
        log(f"[6.1] Loading labeled trips from {path.name}...")
        self.df = pd.read_parquet(path)
        log(f"[6.1] Loaded {len(self.df):,} trips")
        return self

    def engineer_features(self) -> 'TripClassifierPipeline':
        """
        Construct all model features from the labeled dataset.
        Adds new columns to self.df and builds self.X and self.y.
        """
        log("[6.2] Engineering features...")
        df = self.df.copy()

        # ----- Temporal 
        # Already in df: hour, day_of_week, month, is_weekend,
        # is_holiday, is_rush_hour

        # ----- Trip characteristics
        df['duration_log'] = np.log1p(df['duration_min'])

        # ----- Member type 
        df['is_member'] = (df['member_casual'] == 'member').astype(int)

        # ----- Bike type
        df['rideable_electric'] = (df['rideable_type'] == 'electric_bike').astype(int)

        # ----- Boolean columns to int
        for col in ('is_weekend', 'is_holiday', 'is_rush_hour',
                    'start_anchored', 'end_anchored'):
            df[col] = df[col].astype(int)

        # -----Target variable
        df['label'] = (df['trip_purpose'] == 'commuter').astype(int)

        self.df = df

        # Build feature matrix and target vector
        # Drop rows with any missing feature values
        df_model = df.dropna(subset=self.FEATURE_COLS + ['label'])
        self.X = df_model[self.FEATURE_COLS].astype(float)
        self.y = df_model['label']

        log(f"[6.2] Feature matrix: {self.X.shape}")
        log(f"[6.2] Class balance — commuter: {self.y.mean()*100:.1f}%  recreational: {(1-self.y.mean())*100:.1f}%")
        return self

    def split(self, test_size: float = 0.15, val_size: float = 0.15,
              random_state: int = 42):
        """
        Stratified train / validation / test split.
        Returns X_train, X_val, X_test, y_train, y_val, y_test.
        """
        from sklearn.model_selection import train_test_split

        log("[6.3] Splitting data (70/15/15 stratified)...")

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y,
            test_size=test_size + val_size,
            stratify=self.y,
            random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=random_state
        )

        log(f"[6.3] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale(self, X_train, X_val, X_test):
        """
        Fit StandardScaler on training set and transform all splits.
        """
        log("[6.4] Scaling features...")
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc   = self.scaler.transform(X_val)
        X_test_sc  = self.scaler.transform(X_test)
        return X_train_sc, X_val_sc, X_test_sc
