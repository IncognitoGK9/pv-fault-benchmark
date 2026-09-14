"""
target.py
Constructs the two label schemes (binary, tier) from the raw Fault column.
"""
import numpy as np
import config


def binary_target(df):
    """1 = any fault code present, 0 = no-fault."""
    return (df["Fault"] != 0.0).astype(int).values


def tier_target(df):
    """0 = No-Fault, 1 = Constant partial shading, 2 = Severe (uniform/intermittent)."""
    return df["Fault"].map(config.TIER_MAP).astype(int).values


def get_target(df, scheme):
    """Dispatch to binary_target or tier_target by name (config.LABEL_SCHEMES)."""
    if scheme == "binary":
        return binary_target(df)
    elif scheme == "tier":
        return tier_target(df)
    raise ValueError(f"Unknown label scheme: {scheme}")
