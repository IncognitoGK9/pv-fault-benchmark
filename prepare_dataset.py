"""
prepare_dataset.py

Loads the raw Reunion PV electrical-production CSV, computes full-dataset
summary statistics (for the paper's Table II), segments the series into
contiguous same-label runs, caps each run's contribution for computational
tractability, and writes a subsampled, group-labelled CSV used by every
downstream script (features.py, target.py, run_experiment.py).

Every retained row keeps its original `episode_id` (the contiguous segment
it came from) so that GroupKFold splitting in pipeline.py can guarantee no
segment is ever divided across train and test.
"""
import json
import os
import numpy as np
import pandas as pd

import config


def _require_raw_csv():
    """Raise a FileNotFoundError listing every candidate path (config.RAW_CSV_CANDIDATES)
    and how to fix it, if the raw CSV isn't found at any of them."""
    if os.path.exists(config.RAW_CSV):
        return
    lines = [
        f"Could not find '{config.RAW_CSV_NAME}'. Looked in, in order:",
        *[f"  - {c}" for c in config.RAW_CSV_CANDIDATES],
        "",
        "Fix: either copy/move the CSV into this project folder "
        f"(next to {os.path.basename(__file__)}), or open config.py and edit "
        "the _raw_candidates / RAW_CSV lines near the top to point at the "
        "exact path where the file lives on this machine.",
    ]
    raise FileNotFoundError("\n".join(lines))


def load_raw():
    """Load the raw CSV, drop rows missing the label or any raw feature, and
    sort by time. Raises FileNotFoundError (see _require_raw_csv) if the
    file cannot be located.
    """
    _require_raw_csv()
    df = pd.read_csv(config.RAW_CSV, parse_dates=["time"])
    df = df.dropna(subset=["Fault"] + config.RAW_FEATURES).copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def segment(df):
    """Assign a contiguous-run `episode_id` to every row: a new id starts
    each time the fault/no-fault state (`fault_bin`) changes. This is what
    makes each episode a safe, indivisible group for GroupKFold.
    """
    df["fault_bin"] = (df["Fault"] != 0.0).astype(int)
    df["episode_id"] = (df["fault_bin"].diff().fillna(0) != 0).cumsum()
    return df


def cap_and_sample(df, seed=config.RANDOM_SEED):
    """Cap each episode's row count (config.MAX_ROWS_PER_FAULT_SEGMENT /
    MAX_ROWS_PER_NOFAULT_SEGMENT, keeping a random contiguous slice per
    episode) and cap total no-fault volume (config.MAX_NOFAULT_SEGMENTS_ROWS)
    by dropping whole no-fault episodes at random, never partial ones.

    Every one of the 17 fault episodes is always kept in full (subject only
    to the per-episode cap) so the benchmark never loses a fault episode to
    the no-fault-volume cap. Returns (subsampled_df, per-episode metadata list).
    """
    rng = np.random.default_rng(seed)
    kept = []
    seg_info = []
    for eid, g in df.groupby("episode_id"):
        g = g.sort_values("time")
        n = len(g)
        label = int(g["fault_bin"].iloc[0])
        cap = config.MAX_ROWS_PER_FAULT_SEGMENT if label == 1 else config.MAX_ROWS_PER_NOFAULT_SEGMENT
        if n > cap:
            start = int(rng.integers(0, n - cap + 1))
            g = g.iloc[start:start + cap]
        kept.append(g)
        seg_info.append({
            "episode_id": int(eid),
            "label": int(g["fault_bin"].iloc[0]),
            "fault_codes": sorted(set(g["Fault"].unique().tolist())),
            "n_original": int(n),
            "n_kept": int(len(g)),
            "start": str(g["time"].iloc[0]),
            "end": str(g["time"].iloc[-1]),
        })

    kept_df = pd.concat(kept, ignore_index=True)

    # Further cap total no-fault volume by randomly dropping whole no-fault
    # segments (never partial ones, to keep groups intact) until under budget.
    nofault_ids = [s["episode_id"] for s in seg_info if s["label"] == 0]
    rng.shuffle(nofault_ids)
    running = kept_df[kept_df["fault_bin"] == 0].shape[0]
    drop_ids = set()
    for eid in nofault_ids:
        if running <= config.MAX_NOFAULT_SEGMENTS_ROWS:
            break
        seg_len = kept_df[(kept_df["episode_id"] == eid)].shape[0]
        drop_ids.add(eid)
        running -= seg_len

    kept_df = kept_df[~kept_df["episode_id"].isin(drop_ids)].reset_index(drop=True)
    seg_info = [s for s in seg_info if s["episode_id"] not in drop_ids]
    return kept_df, seg_info


def main():
    """Run the full subsampling pipeline: load -> segment -> cap/sample ->
    write data/reunion_subsample.csv and data/subsample_metadata.json
    (the latter holding both the pre-cap full-dataset statistics used in
    the paper's Table II and the per-episode record of what was kept).
    """
    print("Loading raw CSV ...")
    raw = load_raw()
    raw_seg = segment(raw.copy())

    full_stats = {
        "n_rows_raw": int(len(raw)),
        "date_min": str(raw["time"].min()),
        "date_max": str(raw["time"].max()),
        "fault_value_counts": {str(k): int(v) for k, v in raw["Fault"].value_counts().sort_index().items()},
        "n_segments_total": int(raw_seg["episode_id"].nunique()),
        "n_fault_segments": int(raw_seg.groupby("episode_id")["fault_bin"].first().sum()),
    }
    print(json.dumps(full_stats, indent=2))

    print("Segmenting + capping for tractable nested CV ...")
    sub, seg_info = cap_and_sample(raw_seg)

    sub.to_csv(config.SUBSAMPLE_CSV, index=False)

    meta = {
        "full_dataset_stats": full_stats,
        "subsample_n_rows": int(len(sub)),
        "subsample_fault_prevalence": float((sub["fault_bin"] == 1).mean()),
        # Sum rows over every fault code mapped to the same tier. A dict
        # comprehension keyed by tier name would let later codes overwrite
        # earlier ones (it reported 2,000 Severe and 1,600 Constant-Partial
        # instead of 2,800 and 4,000).
        "subsample_tier_counts": {
            name: int(sub["Fault"].map(config.TIER_MAP).map(config.TIER_NAMES).eq(name).sum())
            for name in sorted(set(config.TIER_NAMES[config.TIER_MAP[k]] for k in sub["Fault"].unique()))
        },
        "n_episodes_kept": int(sub["episode_id"].nunique()),
        "n_fault_episodes_kept": int(sub.groupby("episode_id")["fault_bin"].first().sum()),
        "segments": seg_info,
    }
    with open(config.SUBSAMPLE_META, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Subsample written: {len(sub)} rows, "
          f"{meta['subsample_fault_prevalence']:.3%} fault prevalence, "
          f"{meta['n_episodes_kept']} episodes ({meta['n_fault_episodes_kept']} fault).")


if __name__ == "__main__":
    main()
