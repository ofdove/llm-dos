#!/usr/bin/env python3
"""
Build a CIC-IDS-2017 style dataset from CUDA trace summaries for ML-based DoS detection.

One row per request (trace). Features = discriminative metrics (duration, throughput,
volume, IAT, per-op latency/bytes). Label = Normal | DoS.

Usage:
  python build_llm_dos_dataset.py [--max-files N] [--out-dir DIR] [--split 0.2]
  --max-files: limit traces per class (default: all)
  --out-dir: output directory (default: datasets)
  --split: fraction for test set, 0 = no split (default: 0.2)
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def find_project_root() -> Path:
    root = Path.cwd()
    while root != root.parent and not (root / "tracing_output" / "normal").exists():
        root = root.parent
    if not (root / "tracing_output" / "normal").exists():
        root = Path.cwd()
    return root


def list_trace_ids(dir_path: Path) -> list[int]:
    if not dir_path.exists():
        return []
    ids = []
    for f in dir_path.glob("*.csv"):
        try:
            ids.append(int(f.stem))
        except ValueError:
            pass
    return sorted(ids)


def analyze_cuda_trace(csv_path: Path, label: str) -> dict:
    """Summarize one CUDA trace CSV. label is 'normal' or 'dos'."""
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ("timestamp", "latency", "bytes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    start_events = df[df["complete"] != "COMPLETE_ALL"]
    completion_events = df[df["complete"] == "COMPLETE_ALL"]
    total_time = (df["timestamp"].max() - df["timestamp"].min()) if len(df) > 0 else 0
    total_latency = completion_events["latency"].sum() if "latency" in df.columns else 0
    total_bytes = start_events["bytes"].sum() if "bytes" in start_events.columns else 0
    num_kernels = len(start_events[start_events["fname"] == "cuLaunchKernel"])

    df_sorted = df.sort_values("timestamp").dropna(subset=["timestamp"])
    iat_s = df_sorted["timestamp"].diff().dropna()
    iat_s = iat_s[iat_s >= 0]
    iat_mean_s = float(iat_s.mean()) if len(iat_s) > 0 else np.nan
    iat_std_s = float(iat_s.std()) if len(iat_s) > 1 else np.nan
    iat_max_s = float(iat_s.max()) if len(iat_s) > 0 else np.nan
    iat_min_s = float(iat_s.min()) if len(iat_s) > 0 else np.nan

    lat = completion_events["latency"].dropna()
    lat = lat[lat > 0]
    latency_std_ms = float(lat.std()) if len(lat) > 1 else np.nan
    latency_max_ms = float(lat.max()) if len(lat) > 0 else np.nan
    latency_min_ms = float(lat.min()) if len(lat) > 0 else np.nan

    byt = start_events.loc[start_events["bytes"] > 0, "bytes"].dropna()
    bytes_std = float(byt.std()) if len(byt) > 1 else np.nan
    bytes_max = int(byt.max()) if len(byt) > 0 else np.nan
    bytes_min = int(byt.min()) if len(byt) > 0 else np.nan

    return {
        "file_id": int(csv_path.stem),
        "label": label,
        "total_events": len(df),
        "start_events": len(start_events),
        "completion_events": len(completion_events),
        "total_time_seconds": total_time,
        "total_latency_ms": total_latency,
        "num_kernels": num_kernels,
        "total_bytes": total_bytes,
        "iat_mean_s": iat_mean_s,
        "iat_std_s": iat_std_s,
        "iat_max_s": iat_max_s,
        "iat_min_s": iat_min_s,
        "latency_std_ms": latency_std_ms,
        "latency_max_ms": latency_max_ms,
        "latency_min_ms": latency_min_ms,
        "bytes_std": bytes_std,
        "bytes_max": bytes_max,
        "bytes_min": bytes_min,
    }


def load_cuda_summaries(dir_path: Path, label: str, max_files: int | None = None) -> pd.DataFrame:
    ids = list_trace_ids(dir_path)
    if max_files is not None:
        ids = ids[:max_files]
    rows = []
    for fid in ids:
        p = dir_path / f"{fid}.csv"
        if p.exists():
            try:
                rows.append(analyze_cuda_trace(p, label))
            except Exception as e:
                print(f"  Skip {p.name}: {e}")
    return pd.DataFrame(rows)


# CIC-IDS-2017 style: discriminative features for DoS vs Normal (one row per flow/request)
FEATURE_COLUMNS = [
    # Flow duration & volume (like Flow Duration, Total Fwd/Bwd Packets, Total Length)
    "total_time_seconds",
    "total_events",
    "num_kernels",
    "total_bytes",
    "total_latency_ms",
    # Throughput (like Flow Bytes/s, Packets/s)
    "bytes_per_second_MB",
    "events_per_second",
    # Ratio / efficiency
    "latency_per_kernel",
    "latency_per_event",
    "wall_clock_ratio",
    # IAT (like Flow IAT Mean/Std/Max/Min)
    "iat_mean_s",
    "iat_std_s",
    "iat_max_s",
    "iat_min_s",
    # Per-op latency (like Packet Length Std/Max/Min)
    "latency_std_ms",
    "latency_max_ms",
    "latency_min_ms",
    # Per-op bytes
    "bytes_std",
    "bytes_max",
    "bytes_min",
]


def main():
    ap = argparse.ArgumentParser(description="Build LLM DoS dataset (CIC-IDS-2017 style)")
    ap.add_argument("--max-files", type=int, default=None, help="Max traces per class (default: all)")
    ap.add_argument("--out-dir", type=str, default="datasets", help="Output directory")
    ap.add_argument("--split", type=float, default=0.2, help="Test fraction (0 = no split)")
    args = ap.parse_args()

    root = find_project_root()
    cuda_normal = root / "tracing_output" / "normal"
    cuda_dos = root / "tracing_output" / "dos"
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CUDA traces...")
    df_n = load_cuda_summaries(cuda_normal, "normal", max_files=args.max_files)
    df_d = load_cuda_summaries(cuda_dos, "dos", max_files=args.max_files)
    if df_n.empty and df_d.empty:
        print("No traces loaded. Check tracing_output/normal and tracing_output/dos.")
        return
    df = pd.concat([df_n, df_d], ignore_index=True)
    print(f"  normal: {len(df_n)}, dos: {len(df_d)}, total: {len(df)}")

    # Derived columns (same as notebook 8b)
    df["latency_per_kernel"] = df["total_latency_ms"] / df["num_kernels"].replace(0, np.nan)
    df["latency_per_event"] = df["total_latency_ms"] / df["total_events"].replace(0, np.nan)
    df["bytes_per_second_MB"] = (df["total_bytes"] / 1e6) / df["total_time_seconds"].replace(0, np.nan)
    df["events_per_second"] = df["total_events"] / df["total_time_seconds"].replace(0, np.nan)
    df["wall_clock_ratio"] = df["total_latency_ms"] / (df["total_time_seconds"] * 1000).replace(0, np.nan)

    # CIC-IDS-2017 style label: "Normal" / "DoS"
    df["Label"] = df["label"].map({"normal": "Normal", "dos": "DoS"})
    df["label_binary"] = (df["label"] == "dos").astype(int)

    # Build ML dataframe: optional file_id for reproducibility, then features, then Label
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print("Warning: missing feature columns:", missing)
    use_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    ml_df = df[["file_id", "label", "Label", "label_binary"] + use_cols].copy()
    # Drop rows with NaN in too many features (optional: keep and impute later)
    ml_df = ml_df.dropna(subset=use_cols, how="all")
    nan_count = ml_df[use_cols].isna().any(axis=1).sum()
    if nan_count:
        ml_df = ml_df.dropna(subset=use_cols)
        print(f"  Dropped {nan_count} rows with NaN in features; remaining: {len(ml_df)}")

    # Save full dataset
    out_csv = out_dir / "llm_dos_cic_style.csv"
    ml_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  shape={ml_df.shape}")

    if args.split > 0 and args.split < 1:
        rng = np.random.default_rng(9)
        train_dfs = []
        test_dfs = []
        for lb in ml_df["label_binary"].unique():
            sub = ml_df[ml_df["label_binary"] == lb]
            n = len(sub)
            idx = np.arange(n)
            rng.shuffle(idx)
            k = max(1, int(n * args.split))
            test_dfs.append(sub.iloc[idx[:k]])
            train_dfs.append(sub.iloc[idx[k:]])
        test_df = pd.concat(test_dfs, ignore_index=True)
        train_df = pd.concat(train_dfs, ignore_index=True)
        train_csv = out_dir / "llm_dos_train.csv"
        test_csv = out_dir / "llm_dos_test.csv"
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        print(f"Split ({1-args.split:.0%} train / {args.split:.0%} test):")
        print(f"  {train_csv}  shape={train_df.shape}")
        print(f"  {test_csv}  shape={test_df.shape}")

    print("Label counts:", ml_df["Label"].value_counts().to_dict())
    print("Feature columns:", use_cols)


if __name__ == "__main__":
    main()
