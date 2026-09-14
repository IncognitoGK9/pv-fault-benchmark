"""
features.py
Feature engineering on the electrical channels of the Reunion PV dataset.

All rolling/diff features are computed WITHIN episode_id groups only, so no
engineered feature ever looks across an episode boundary (which would
otherwise leak information between an episode and its neighbours).
"""
import numpy as np
import pandas as pd

import config


def add_derived_columns(df):
    """Add four instantaneous, physically-motivated ratio/deviation features
    (current ratio, power-conversion-efficiency ratio, frequency deviation,
    voltage drop across the inverter stage) computed from a single row.
    """
    df = df.copy()
    eps = 1e-6
    df["ratio_Ig_Ia"] = df["Ig"] / (df["Ia"] + eps)
    df["ratio_Pg_VaIa"] = df["Pg"] / (df["Va"] * df["Ia"] + eps)
    df["freq_dev"] = (df["Fg"] - 50.0).abs()
    df["volt_dev"] = df["Vg"] - df["Va"]
    return df


def add_rolling_features(df):
    """Add rolling mean/std (over config.ROLLING_WINDOWS) and first-difference
    features for the core electrical channels, grouped by episode_id so no
    window or difference ever spans an episode boundary.
    """
    df = df.sort_values(["episode_id", "time"]).copy()
    group = df.groupby("episode_id")
    for col in ["Pg", "Ia", "Vg", "Fg"]:
        for w in config.ROLLING_WINDOWS:
            df[f"{col}_rmean{w}"] = group[col].transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean())
            df[f"{col}_rstd{w}"] = group[col].transform(
                lambda s, w=w: s.rolling(w, min_periods=1).std().fillna(0.0))
    for col in ["Pg", "Ia"]:
        df[f"{col}_diff1"] = group[col].diff().fillna(0.0)
    return df


def build_features(df):
    """Build the full 29-column feature matrix (7 raw channels + 4 derived +
    16 rolling + 2 rate-of-change) used throughout the experiment.

    Returns (X, feature_cols): X is the feature DataFrame with inf/NaN
    replaced by 0.0, feature_cols is the ordered list of its column names.
    """
    df = add_derived_columns(df)
    df = add_rolling_features(df)
    feature_cols = list(config.RAW_FEATURES) + [
        "ratio_Ig_Ia", "ratio_Pg_VaIa", "freq_dev", "volt_dev",
    ]
    for col in ["Pg", "Ia", "Vg", "Fg"]:
        for w in config.ROLLING_WINDOWS:
            feature_cols += [f"{col}_rmean{w}", f"{col}_rstd{w}"]
    feature_cols += ["Pg_diff1", "Ia_diff1"]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, feature_cols
