#!/usr/bin/env python3
"""
Train and evaluate classifiers for LLM DoS detection.

Classifiers: Random Forest, Gradient Boosting, Logistic Regression
(as proposed in the GLOBECOM paper).

Usage:
  uv run python train_classifiers.py
  uv run python train_classifiers.py --out-dir models
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURE_COLUMNS = [
    "total_time_seconds",
    "total_events",
    "num_kernels",
    "total_bytes",
    "total_latency_ms",
    "bytes_per_second_MB",
    "events_per_second",
    "latency_per_kernel",
    "latency_per_event",
    "wall_clock_ratio",
    "iat_mean_s",
    "iat_std_s",
    "iat_max_s",
    "iat_min_s",
    "latency_std_ms",
    "latency_max_ms",
    "latency_min_ms",
    "bytes_std",
    "bytes_max",
    "bytes_min",
]


def load_data(datasets_dir: Path):
    train_path = datasets_dir / "llm_dos_train.csv"
    test_path = datasets_dir / "llm_dos_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path} or {test_path}. "
            "Run: uv run python build_llm_dos_dataset.py --split 0.2"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    use_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]

    X_train = train_df[use_cols].values
    y_train = train_df["label_binary"].values
    X_test = test_df[use_cols].values
    y_test = test_df["label_binary"].values

    return X_train, y_train, X_test, y_test, use_cols, train_df, test_df


def evaluate(name: str, clf, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    return {
        "classifier": name,
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "fpr": round(fpr, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def print_results_table(results: list[dict]):
    header = f"{'Classifier':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'FPR':>6} {'AUC':>6}  {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        auc_str = f"{r['auc']:.4f}" if r["auc"] is not None else "  N/A"
        print(
            f"{r['classifier']:<25} {r['f1']:>6.4f} {r['precision']:>6.4f} "
            f"{r['recall']:>6.4f} {r['fpr']:>6.4f} {auc_str:>6}  "
            f"{r['tp']:>4} {r['fp']:>4} {r['tn']:>4} {r['fn']:>4}"
        )
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser(description="Train LLM DoS classifiers")
    ap.add_argument("--datasets-dir", type=str, default="datasets")
    ap.add_argument("--out-dir", type=str, default="models")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    datasets_dir = root / args.datasets_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_train, y_train, X_test, y_test, feature_names, train_df, test_df = load_data(datasets_dir)
    print(f"  Train: {X_train.shape[0]} samples ({(y_train == 0).sum()} normal, {(y_train == 1).sum()} dos)")
    print(f"  Test:  {X_test.shape[0]} samples ({(y_test == 0).sum()} normal, {(y_test == 1).sum()} dos)")
    print(f"  Features: {len(feature_names)}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    }

    results = []
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        if name == "Logistic Regression":
            clf.fit(X_train_s, y_train)
            r = evaluate(name, clf, X_test_s, y_test)
        else:
            clf.fit(X_train, y_train)
            r = evaluate(name, clf, X_test, y_test)
        results.append(r)

        print(f"  F1={r['f1']:.4f}  Precision={r['precision']:.4f}  Recall={r['recall']:.4f}  FPR={r['fpr']:.4f}")
        print(f"  Confusion: TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")

        if name == "Logistic Regression":
            print("\n  Classification Report:")
            y_pred = clf.predict(X_test_s)
        else:
            print("\n  Classification Report:")
            y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["Normal", "DoS"], digits=4))

    print_results_table(results)

    # Feature importance (Random Forest)
    rf = classifiers["Random Forest"]
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nRandom Forest Feature Importance (top 10):")
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"  {i+1}. {feature_names[idx]:<25} {importances[idx]:.4f}")

    # Cohen's d for all features
    print("\nCohen's d (effect size) for each feature:")
    cohens_d = []
    for feat in feature_names:
        col_idx = feature_names.index(feat)
        n_vals = X_train[y_train == 0, col_idx]
        d_vals = X_train[y_train == 1, col_idx]
        n_vals = n_vals[~np.isnan(n_vals)]
        d_vals = d_vals[~np.isnan(d_vals)]
        if len(n_vals) < 2 or len(d_vals) < 2:
            continue
        pooled_std = np.sqrt((n_vals.var() + d_vals.var()) / 2)
        if pooled_std == 0:
            continue
        d = (d_vals.mean() - n_vals.mean()) / pooled_std
        cohens_d.append((feat, d))
    cohens_d.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, d in cohens_d:
        print(f"  {feat:<25} d = {d:+.4f}")

    # Save models and metadata
    joblib.dump(rf, out_dir / "random_forest.joblib")
    joblib.dump(classifiers["Gradient Boosting"], out_dir / "gradient_boosting.joblib")
    joblib.dump(classifiers["Logistic Regression"], out_dir / "logistic_regression.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")

    metadata = {
        "feature_columns": feature_names,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "results": results,
        "feature_importance_top10": [
            {"feature": feature_names[idx], "importance": round(float(importances[idx]), 4)}
            for idx in sorted_idx[:10]
        ],
        "cohens_d": [{"feature": f, "d": round(d, 4)} for f, d in cohens_d],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModels saved to {out_dir}/")
    print(f"  random_forest.joblib, gradient_boosting.joblib, logistic_regression.joblib")
    print(f"  scaler.joblib, metadata.json")


if __name__ == "__main__":
    main()
