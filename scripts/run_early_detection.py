#!/usr/bin/env python3
"""
Early detection experiment: truncate each trace to the first T seconds,
re-extract features, and evaluate classification accuracy.

Tests T = 1, 2, 3, 5, 10, 15 seconds.

Usage:
  uv run python run_early_detection.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "qian" / "26globecom" / "ieee-enhanced-main" / "figures"
EXP_DIR = ROOT / "experiments"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

TIME_WINDOWS = [1, 2, 3, 5, 10, 15]


def analyze_truncated_trace(csv_path: Path, max_duration_s: float) -> dict | None:
    """Analyze a trace truncated to the first max_duration_s seconds."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None

    for col in ("timestamp", "latency", "bytes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) == 0 or "timestamp" not in df.columns:
        return None

    t0 = df["timestamp"].min()
    if pd.isna(t0):
        return None

    df = df[df["timestamp"] <= t0 + max_duration_s]
    if len(df) < 10:
        return None

    start_events = df[df["complete"] != "COMPLETE_ALL"]
    completion_events = df[df["complete"] == "COMPLETE_ALL"]

    total_time = df["timestamp"].max() - t0
    if total_time <= 0:
        return None

    total_latency = completion_events["latency"].sum() if "latency" in df.columns else 0
    total_bytes = start_events["bytes"].sum() if "bytes" in start_events.columns else 0
    num_kernels = len(start_events[start_events["fname"] == "cuLaunchKernel"])

    df_sorted = df.sort_values("timestamp").dropna(subset=["timestamp"])
    iat_s = df_sorted["timestamp"].diff().dropna()
    iat_s = iat_s[iat_s >= 0]

    lat = completion_events["latency"].dropna()
    lat = lat[lat > 0]

    byt = start_events.loc[start_events["bytes"] > 0, "bytes"].dropna()

    return {
        "total_time_seconds": total_time,
        "total_events": len(df),
        "num_kernels": num_kernels,
        "total_bytes": float(total_bytes),
        "total_latency_ms": float(total_latency),
        "bytes_per_second_MB": (float(total_bytes) / 1e6) / total_time if total_time > 0 else 0,
        "events_per_second": len(df) / total_time if total_time > 0 else 0,
        "latency_per_kernel": float(total_latency) / num_kernels if num_kernels > 0 else np.nan,
        "latency_per_event": float(total_latency) / len(df) if len(df) > 0 else np.nan,
        "wall_clock_ratio": float(total_latency) / (total_time * 1000) if total_time > 0 else np.nan,
        "iat_mean_s": float(iat_s.mean()) if len(iat_s) > 0 else np.nan,
        "iat_std_s": float(iat_s.std()) if len(iat_s) > 1 else np.nan,
        "iat_max_s": float(iat_s.max()) if len(iat_s) > 0 else np.nan,
        "iat_min_s": float(iat_s.min()) if len(iat_s) > 0 else np.nan,
        "latency_std_ms": float(lat.std()) if len(lat) > 1 else np.nan,
        "latency_max_ms": float(lat.max()) if len(lat) > 0 else np.nan,
        "latency_min_ms": float(lat.min()) if len(lat) > 0 else np.nan,
        "bytes_std": float(byt.std()) if len(byt) > 1 else np.nan,
        "bytes_max": float(byt.max()) if len(byt) > 0 else np.nan,
        "bytes_min": float(byt.min()) if len(byt) > 0 else np.nan,
    }


FEATURE_COLUMNS = [
    "total_time_seconds", "total_events", "num_kernels", "total_bytes",
    "total_latency_ms", "bytes_per_second_MB", "events_per_second",
    "latency_per_kernel", "latency_per_event", "wall_clock_ratio",
    "iat_mean_s", "iat_std_s", "iat_max_s", "iat_min_s",
    "latency_std_ms", "latency_max_ms", "latency_min_ms",
    "bytes_std", "bytes_max", "bytes_min",
]


def main():
    normal_dir = ROOT / "tracing_output" / "normal"
    dos_dir = ROOT / "tracing_output" / "dos"

    # Use the same test split file IDs
    test_df = pd.read_csv(ROOT / "datasets" / "llm_dos_test.csv")
    train_df = pd.read_csv(ROOT / "datasets" / "llm_dos_train.csv")

    test_normal_ids = test_df[test_df["label"] == "normal"]["file_id"].tolist()
    test_dos_ids = test_df[test_df["label"] == "dos"]["file_id"].tolist()

    # Train on full traces (use existing train set)
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["label_binary"].values

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Trained RF on {len(X_train)} full-trace samples")

    # For each time window, truncate test traces and evaluate
    results = {}
    for T in TIME_WINDOWS:
        print(f"\n--- T = {T}s ---")
        rows = []
        skipped = 0

        for fid in test_normal_ids:
            p = normal_dir / f"{fid}.csv"
            if not p.exists():
                skipped += 1
                continue
            r = analyze_truncated_trace(p, T)
            if r is not None:
                r["label_binary"] = 0
                rows.append(r)
            else:
                skipped += 1

        for fid in test_dos_ids:
            p = dos_dir / f"{fid}.csv"
            if not p.exists():
                skipped += 1
                continue
            r = analyze_truncated_trace(p, T)
            if r is not None:
                r["label_binary"] = 1
                rows.append(r)
            else:
                skipped += 1

        if not rows:
            print(f"  No valid traces for T={T}s")
            continue

        df_t = pd.DataFrame(rows)
        df_t = df_t.dropna(subset=FEATURE_COLUMNS)

        X_test_t = df_t[FEATURE_COLUMNS].values
        y_test_t = df_t["label_binary"].values

        y_pred = rf.predict(X_test_t)
        y_prob = rf.predict_proba(X_test_t)[:, 1]

        f1 = f1_score(y_test_t, y_pred)
        prec = precision_score(y_test_t, y_pred)
        rec = recall_score(y_test_t, y_pred)
        auc_val = roc_auc_score(y_test_t, y_prob)

        n_normal = (y_test_t == 0).sum()
        n_dos = (y_test_t == 1).sum()
        fp = ((y_pred == 1) & (y_test_t == 0)).sum()
        fpr = fp / n_normal if n_normal > 0 else 0

        results[f"T={T}s"] = {
            "window_seconds": T,
            "n_samples": len(df_t),
            "n_normal": int(n_normal),
            "n_dos": int(n_dos),
            "skipped": skipped,
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "fpr": round(fpr, 4),
            "auc": round(auc_val, 4),
        }

        print(f"  Samples: {len(df_t)} (normal={n_normal}, dos={n_dos}, skipped={skipped})")
        print(f"  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  FPR={fpr:.4f}  AUC={auc_val:.4f}")

    # Save results
    out = EXP_DIR / "early_detection.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")

    # Generate figure
    if results:
        windows = [results[k]["window_seconds"] for k in results]
        f1s = [results[k]["f1"] for k in results]
        precs = [results[k]["precision"] for k in results]
        recs = [results[k]["recall"] for k in results]

        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(windows, f1s, "o-", color="#2166ac", linewidth=1.5, markersize=5, label="F1")
        ax.plot(windows, precs, "s--", color="#4daf4a", linewidth=1.2, markersize=4, label="Precision")
        ax.plot(windows, recs, "^--", color="#b2182b", linewidth=1.2, markersize=4, label="Recall")

        ax.set_xlabel("Detection window $T$ (seconds)", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(0.5, 1.02)
        ax.set_xticks(windows)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "early_detection.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {FIG_DIR / 'early_detection.pdf'}")


if __name__ == "__main__":
    main()
