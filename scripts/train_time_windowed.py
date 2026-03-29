#!/usr/bin/env python3
"""
Train and evaluate classifiers at different time windows (T seconds from start).

For each T, truncates both CUDA and CPU traces to the first T seconds,
re-extracts features, and trains/evaluates separate models.

Compares GPU-only, CPU-only, and GPU+CPU at each window.

Usage:
  uv run python train_time_windowed.py
  uv run python train_time_windowed.py --windows 1 2 3 5 10
"""

from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


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


GPU_FEATURES = [
    "total_time_seconds", "total_events", "num_kernels", "total_bytes",
    "total_latency_ms", "bytes_per_second_MB", "events_per_second",
    "latency_per_kernel", "latency_per_event", "wall_clock_ratio",
    "iat_mean_s", "iat_std_s", "iat_max_s", "iat_min_s",
    "latency_std_ms", "latency_max_ms", "latency_min_ms",
    "bytes_std", "bytes_max", "bytes_min",
]

CPU_FEATURES = [
    "cpu_mmap_count", "cpu_mmap_dur_mean_ns",
    "cpu_sched_lat_mean_ns", "cpu_sched_lat_std_ns", "cpu_offcpu_dur_max_ns",
    "cpu_futex_dur_mean_ns", "cpu_futex_dur_p95_ns", "cpu_futex_per_sec",
    "cpu_unique_tids", "cpu_unique_comms",
]


def extract_cuda_features(csv_path: Path, max_time_s: float | None = None) -> dict | None:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None
    for col in ("timestamp", "latency", "bytes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) < 2 or "timestamp" not in df.columns:
        return None

    t0 = df["timestamp"].min()
    if max_time_s is not None:
        df = df[df["timestamp"] <= t0 + max_time_s]
    if len(df) < 2:
        return None

    total_time = df["timestamp"].max() - t0
    if total_time <= 0:
        return None

    start_events = df[df["complete"] != "COMPLETE_ALL"]
    completion_events = df[df["complete"] == "COMPLETE_ALL"]
    total_latency = completion_events["latency"].sum() if "latency" in df.columns else 0
    total_bytes = start_events["bytes"].sum() if "bytes" in start_events.columns else 0
    num_kernels = len(start_events[start_events["fname"] == "cuLaunchKernel"])
    n_events = len(df)

    df_sorted = df.sort_values("timestamp").dropna(subset=["timestamp"])
    iat_s = df_sorted["timestamp"].diff().dropna()
    iat_s = iat_s[iat_s >= 0]

    lat = completion_events["latency"].dropna()
    lat = lat[lat > 0]

    byt = start_events.loc[start_events["bytes"] > 0, "bytes"].dropna()

    feat = {
        "total_time_seconds": total_time,
        "total_events": n_events,
        "num_kernels": num_kernels,
        "total_bytes": total_bytes,
        "total_latency_ms": total_latency,
        "bytes_per_second_MB": (total_bytes / 1e6) / total_time if total_time > 0 else 0,
        "events_per_second": n_events / total_time if total_time > 0 else 0,
        "latency_per_kernel": total_latency / num_kernels if num_kernels > 0 else np.nan,
        "latency_per_event": total_latency / n_events if n_events > 0 else np.nan,
        "wall_clock_ratio": total_latency / (total_time * 1000) if total_time > 0 else np.nan,
        "iat_mean_s": float(iat_s.mean()) if len(iat_s) > 0 else np.nan,
        "iat_std_s": float(iat_s.std()) if len(iat_s) > 1 else np.nan,
        "iat_max_s": float(iat_s.max()) if len(iat_s) > 0 else np.nan,
        "iat_min_s": float(iat_s.min()) if len(iat_s) > 0 else np.nan,
        "latency_std_ms": float(lat.std()) if len(lat) > 1 else np.nan,
        "latency_max_ms": float(lat.max()) if len(lat) > 0 else np.nan,
        "latency_min_ms": float(lat.min()) if len(lat) > 0 else np.nan,
        "bytes_std": float(byt.std()) if len(byt) > 1 else np.nan,
        "bytes_max": int(byt.max()) if len(byt) > 0 else np.nan,
        "bytes_min": int(byt.min()) if len(byt) > 0 else np.nan,
    }
    return feat


def extract_cpu_features(csv_path: Path, max_time_s: float | None = None) -> dict | None:
    try:
        df = pd.read_csv(csv_path, low_memory=False,
                         usecols=["timestamp", "fname", "dur_ns", "tid", "comm"])
    except Exception:
        return None
    if "timestamp" not in df.columns or len(df) < 10:
        return None

    df["dur_ns"] = pd.to_numeric(df["dur_ns"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    t0 = df["timestamp"].min()
    if max_time_s is not None:
        df = df[df["timestamp"] <= t0 + max_time_s]
    if len(df) < 10:
        return None

    total_time = df["timestamp"].max() - t0
    if total_time <= 0:
        return None

    mmap = df.loc[df["fname"] == "CPU_MMAP", "dur_ns"].dropna()
    offcpu = df.loc[df["fname"] == "CPU_OFFCPU", "dur_ns"].dropna()
    futex = df.loc[df["fname"] == "CPU_FUTEX_WAIT", "dur_ns"].dropna()

    feat = {
        "cpu_mmap_count": len(mmap),
        "cpu_mmap_dur_mean_ns": mmap.mean() if len(mmap) > 0 else 0,
        "cpu_sched_lat_mean_ns": offcpu.mean() if len(offcpu) > 0 else 0,
        "cpu_sched_lat_std_ns": offcpu.std() if len(offcpu) > 1 else 0,
        "cpu_offcpu_dur_max_ns": offcpu.max() if len(offcpu) > 0 else 0,
        "cpu_futex_dur_mean_ns": futex.mean() if len(futex) > 0 else 0,
        "cpu_futex_dur_p95_ns": futex.quantile(0.95) if len(futex) > 1 else 0,
        "cpu_futex_per_sec": len(futex) / total_time if total_time > 0 else 0,
        "cpu_unique_tids": df["tid"].nunique(),
        "cpu_unique_comms": df["comm"].nunique(),
    }
    return feat


def build_dataset_at_window(root: Path, max_time_s: float | None, max_files: int | None = None):
    rows = []
    for label_dir, label_val in [("normal", 0), ("dos", 1)]:
        cuda_dir = root / "tracing_output" / label_dir
        cpu_dir = root / "tracing_output_cpu" / label_dir
        ids = list_trace_ids(cuda_dir)
        if max_files:
            ids = ids[:max_files]
        for fid in ids:
            cuda_path = cuda_dir / f"{fid}.csv"
            cpu_path = cpu_dir / f"{fid}.csv"
            if not cuda_path.exists():
                continue
            gpu_feat = extract_cuda_features(cuda_path, max_time_s)
            if gpu_feat is None:
                continue
            row = {"file_id": fid, "label": label_val}
            row.update(gpu_feat)
            if cpu_path.exists():
                cpu_feat = extract_cpu_features(cpu_path, max_time_s)
                if cpu_feat:
                    row.update(cpu_feat)
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_rf(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]

    tn = sum((y_pred == 0) & (y_test == 0))
    fp = sum((y_pred == 1) & (y_test == 0))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "fpr": round(fpr, 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def run_cv(df, feature_cols, n_folds=5):
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    y = df["label"].values

    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy="median").fit_transform(X)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    for train_idx, test_idx in skf.split(X, y):
        res = evaluate_rf(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        results.append(res)

    return {
        "f1": round(np.mean([r["f1"] for r in results]), 4),
        "f1_std": round(np.std([r["f1"] for r in results]), 4),
        "precision": round(np.mean([r["precision"] for r in results]), 4),
        "recall": round(np.mean([r["recall"] for r in results]), 4),
        "fpr": round(np.mean([r["fpr"] for r in results]), 4),
        "auc": round(np.mean([r["auc"] for r in results]), 4),
        "n_features": len(available),
        "n_samples": len(df),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", type=float,
                    default=[1, 2, 3, 4, 5, 10, 15],
                    help="Time windows in seconds")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default="experiments")
    args = ap.parse_args()

    root = find_project_root()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = args.windows + [None]  # None = full trace
    all_results = []

    for T in windows:
        label = f"T={T:.0f}s" if T is not None else "Full"
        print(f"\n{'='*70}")
        print(f"  Window: {label}")
        print(f"{'='*70}")
        t0 = time.time()

        df = build_dataset_at_window(root, T, max_files=args.max_files)
        n_normal = sum(df.label == 0)
        n_dos = sum(df.label == 1)
        elapsed = time.time() - t0
        print(f"  Dataset: {len(df)} samples (Normal={n_normal}, DoS={n_dos}) [{elapsed:.0f}s]")

        if len(df) < 20:
            print("  SKIP: too few samples")
            continue

        has_cpu = all(c in df.columns for c in CPU_FEATURES[:2])

        for mode_name, feat_cols in [
            ("GPU-only", GPU_FEATURES),
            ("CPU-only", CPU_FEATURES),
            ("GPU+CPU", GPU_FEATURES + CPU_FEATURES),
        ]:
            if mode_name in ("CPU-only", "GPU+CPU") and not has_cpu:
                print(f"  {mode_name}: SKIP (no CPU traces)")
                continue
            avail = [c for c in feat_cols if c in df.columns]
            sub = df.dropna(subset=avail, how="all")
            if len(sub) < 20:
                print(f"  {mode_name}: SKIP (too few valid samples)")
                continue

            res = run_cv(sub, avail)
            res["window"] = label
            res["mode"] = mode_name
            all_results.append(res)
            print(f"  {mode_name:10s}: F1={res['f1']:.3f}±{res['f1_std']:.3f}  "
                  f"Prec={res['precision']:.3f}  Rec={res['recall']:.3f}  "
                  f"FPR={res['fpr']:.3f}  AUC={res['auc']:.3f}  "
                  f"({res['n_features']} feats, {res['n_samples']} samples)")

    results_df = pd.DataFrame(all_results)
    out_path = out_dir / "time_windowed_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    print(f"{'Window':<8s} {'Mode':<12s} {'F1':>8s} {'±std':>7s} {'Prec':>8s} "
          f"{'Recall':>8s} {'FPR':>8s} {'AUC':>8s} {'#Feat':>6s}")
    print("-"*90)
    for r in all_results:
        print(f"{r['window']:<8s} {r['mode']:<12s} {r['f1']:>8.3f} {r['f1_std']:>7.3f} "
              f"{r['precision']:>8.3f} {r['recall']:>8.3f} {r['fpr']:>8.3f} "
              f"{r['auc']:>8.3f} {r['n_features']:>6d}")


if __name__ == "__main__":
    main()
