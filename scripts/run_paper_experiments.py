#!/usr/bin/env python3
"""
Run all experiments and generate all figures for the GLOBECOM paper.

Outputs:
  paper/ieee-enhanced-main/figures/cdf_duration.pdf
  paper/ieee-enhanced-main/figures/feature_importance.pdf
  paper/ieee-enhanced-main/figures/roc_curve.pdf
  paper/ieee-enhanced-main/figures/confusion_matrix.pdf
  experiments/cross_validation.json
  experiments/feature_ablation.json

Usage:
  uv run python run_paper_experiments.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "paper" / "ieee-enhanced-main" / "figures"
EXP_DIR = ROOT / "experiments"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "total_time_seconds", "total_events", "num_kernels", "total_bytes",
    "total_latency_ms", "bytes_per_second_MB", "events_per_second",
    "latency_per_kernel", "latency_per_event", "wall_clock_ratio",
    "iat_mean_s", "iat_std_s", "iat_max_s", "iat_min_s",
    "latency_std_ms", "latency_max_ms", "latency_min_ms",
    "bytes_std", "bytes_max", "bytes_min",
]

FEATURE_GROUPS = {
    "Duration & Volume": ["total_time_seconds", "total_events", "num_kernels", "total_bytes", "total_latency_ms"],
    "Throughput": ["bytes_per_second_MB", "events_per_second"],
    "Efficiency": ["latency_per_kernel", "latency_per_event", "wall_clock_ratio"],
    "Inter-event Time": ["iat_mean_s", "iat_std_s", "iat_max_s", "iat_min_s"],
    "Per-op Latency": ["latency_std_ms", "latency_max_ms", "latency_min_ms"],
    "Per-op Bytes": ["bytes_std", "bytes_max", "bytes_min"],
}


def load_data():
    full = pd.read_csv(ROOT / "datasets" / "llm_dos_cic_style.csv")
    train = pd.read_csv(ROOT / "datasets" / "llm_dos_train.csv")
    test = pd.read_csv(ROOT / "datasets" / "llm_dos_test.csv")
    return full, train, test


# ──────────────────────────────────────────────────────────────
# Figure 1: CDF of request duration
# ──────────────────────────────────────────────────────────────
def fig_cdf(full: pd.DataFrame):
    print("[Fig] CDF of request duration...")
    fig, ax = plt.subplots(figsize=(5, 3.2))

    for label, color, ls in [("normal", "#2166ac", "-"), ("dos", "#b2182b", "--")]:
        sub = full[full["label"] == label]["total_time_seconds"].dropna().sort_values()
        y = np.linspace(0, 1, len(sub), endpoint=True)
        display = "Normal" if label == "normal" else "DoS"
        ax.plot(sub.values, y, label=display, color=color, linewidth=1.5, linestyle=ls)

    ax.set_xlabel("Request duration (seconds)", fontsize=9)
    ax.set_ylabel("Cumulative fraction", fontsize=9)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cdf_duration.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'cdf_duration.pdf'}")


# ──────────────────────────────────────────────────────────────
# Figure 2: Feature importance (Random Forest)
# ──────────────────────────────────────────────────────────────
def fig_feature_importance(train: pd.DataFrame):
    print("[Fig] Feature importance...")
    X = train[FEATURE_COLUMNS].values
    y = train["label_binary"].values
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:10]

    short_names = {
        "total_time_seconds": "total_time",
        "total_events": "total_events",
        "num_kernels": "num_kernels",
        "total_bytes": "total_bytes",
        "total_latency_ms": "total_latency",
        "bytes_per_second_MB": "bytes/sec",
        "events_per_second": "events/sec",
        "latency_per_kernel": "lat/kernel",
        "latency_per_event": "lat/event",
        "wall_clock_ratio": "wall_ratio",
        "iat_mean_s": "iat_mean",
        "iat_std_s": "iat_std",
        "iat_max_s": "iat_max",
        "iat_min_s": "iat_min",
        "latency_std_ms": "lat_std",
        "latency_max_ms": "lat_max",
        "latency_min_ms": "lat_min",
        "bytes_std": "bytes_std",
        "bytes_max": "bytes_max",
        "bytes_min": "bytes_min",
    }

    names = [short_names.get(FEATURE_COLUMNS[i], FEATURE_COLUMNS[i]) for i in idx]
    values = imp[idx]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.barh(range(len(names)), values, color="#2166ac", edgecolor="white", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance (MDI)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.3)

    for bar, v in zip(bars, values):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'feature_importance.pdf'}")


# ──────────────────────────────────────────────────────────────
# Figure 3: ROC curves
# ──────────────────────────────────────────────────────────────
def fig_roc(train: pd.DataFrame, test: pd.DataFrame):
    print("[Fig] ROC curves...")
    X_train = train[FEATURE_COLUMNS].values
    y_train = train["label_binary"].values
    X_test = test[FEATURE_COLUMNS].values
    y_test = test["label_binary"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X_train, X_test),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42), X_train, X_test),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), X_train_s, X_test_s),
    ]

    colors = ["#2166ac", "#b2182b", "#4daf4a"]
    fig, ax = plt.subplots(figsize=(5, 4))

    for (name, clf, Xtr, Xte), color in zip(classifiers, colors):
        clf.fit(Xtr, y_train)
        y_prob = clf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.5,
                label=f"{name} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'roc_curve.pdf'}")


# ──────────────────────────────────────────────────────────────
# Figure 4: Confusion matrix (Random Forest)
# ──────────────────────────────────────────────────────────────
def fig_confusion(train: pd.DataFrame, test: pd.DataFrame):
    print("[Fig] Confusion matrix...")
    X_train = train[FEATURE_COLUMNS].values
    y_train = train["label_binary"].values
    X_test = test[FEATURE_COLUMNS].values
    y_test = test["label_binary"].values

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(3.5, 3))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "DoS"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'confusion_matrix.pdf'}")


# ──────────────────────────────────────────────────────────────
# Experiment 1: 5-fold cross-validation
# ──────────────────────────────────────────────────────────────
def run_cross_validation(full: pd.DataFrame):
    print("\n[Exp] 5-fold cross-validation...")
    X = full[FEATURE_COLUMNS].values
    y = full["label_binary"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    classifiers = {
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42), X),
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), X_s),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["f1", "precision", "recall", "roc_auc"]

    results = {}
    for name, (clf, X_use) in classifiers.items():
        scores = cross_validate(clf, X_use, y, cv=cv, scoring=scoring, n_jobs=-1)
        r = {}
        for metric in scoring:
            vals = scores[f"test_{metric}"]
            r[metric] = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)}
            print(f"  {name} {metric}: {vals.mean():.4f} ± {vals.std():.4f}")
        results[name] = r

    out = EXP_DIR / "cross_validation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out}")
    return results


# ──────────────────────────────────────────────────────────────
# Experiment 2: Feature ablation study
# ──────────────────────────────────────────────────────────────
def run_ablation(full: pd.DataFrame):
    print("\n[Exp] Feature ablation study...")
    X_all = full[FEATURE_COLUMNS].values
    y = full["label_binary"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # Full model baseline
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_validate(rf_full, X_all, y, cv=cv, scoring=["f1", "roc_auc"], n_jobs=-1)
    results["All features (20)"] = {
        "features": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "f1_mean": round(float(scores["test_f1"].mean()), 4),
        "f1_std": round(float(scores["test_f1"].std()), 4),
        "auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
        "auc_std": round(float(scores["test_roc_auc"].std()), 4),
    }
    print(f"  All features: F1={scores['test_f1'].mean():.4f}±{scores['test_f1'].std():.4f}")

    # Each group alone
    for group_name, group_cols in FEATURE_GROUPS.items():
        avail = [c for c in group_cols if c in full.columns]
        if not avail:
            continue
        X_g = full[avail].values
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores = cross_validate(rf, X_g, y, cv=cv, scoring=["f1", "roc_auc"], n_jobs=-1)
        results[group_name] = {
            "features": avail,
            "n_features": len(avail),
            "f1_mean": round(float(scores["test_f1"].mean()), 4),
            "f1_std": round(float(scores["test_f1"].std()), 4),
            "auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
            "auc_std": round(float(scores["test_roc_auc"].std()), 4),
        }
        print(f"  {group_name} only ({len(avail)} features): F1={scores['test_f1'].mean():.4f}±{scores['test_f1'].std():.4f}")

    # Leave-one-group-out
    print("  --- Leave-one-group-out ---")
    for group_name, group_cols in FEATURE_GROUPS.items():
        remaining = [c for c in FEATURE_COLUMNS if c not in group_cols and c in full.columns]
        if not remaining:
            continue
        X_r = full[remaining].values
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores = cross_validate(rf, X_r, y, cv=cv, scoring=["f1", "roc_auc"], n_jobs=-1)
        key = f"Without {group_name}"
        results[key] = {
            "features": remaining,
            "n_features": len(remaining),
            "f1_mean": round(float(scores["test_f1"].mean()), 4),
            "f1_std": round(float(scores["test_f1"].std()), 4),
            "auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
            "auc_std": round(float(scores["test_roc_auc"].std()), 4),
        }
        print(f"  Without {group_name}: F1={scores['test_f1'].mean():.4f}±{scores['test_f1'].std():.4f}")

    out = EXP_DIR / "feature_ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out}")
    return results


if __name__ == "__main__":
    full, train, test = load_data()
    print(f"Dataset: {len(full)} total, {len(train)} train, {len(test)} test\n")

    # Figures
    fig_cdf(full)
    fig_feature_importance(train)
    fig_roc(train, test)
    fig_confusion(train, test)

    # Experiments
    cv_results = run_cross_validation(full)
    ablation_results = run_ablation(full)

    print("\n✓ All figures and experiments complete.")
