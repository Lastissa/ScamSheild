"""
step4_train_model.py - Step 4: build the full model pipeline.

Candidates (each calibrated with CalibratedClassifierCV so the score is a
meaningful probability):
  1. logistic regression  (word TF-IDF + char TF-IDF + link features)
  2. multinomial naive Bayes
  3. linear SVM

All are trained on the `text` of the train split only. Link features are
computed INSIDE the pipeline from the raw text via features.LinkFeatureExtractor
(link_tools under the hood).

Candidates are compared on the VALIDATION split by PR-AUC and false-alarm rate.
The best one becomes the final model; thresholds (Step 6) are then chosen on
validation, and full evaluation happens in Step 5.

Outputs:
  model/artifacts/scam_model.joblib   the fitted Pipeline (raw text -> score)
  model/artifacts/model_meta.json     candidate comparison + thresholds (feeds model_card.json)

Run from model/:  python src/step4_train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, PROCESSED_DIR, ARTIFACTS_DIR, load_csv
from features import LinkFeatureExtractor

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature union: raw text -> word TF-IDF + char TF-IDF + link features
# ---------------------------------------------------------------------------
def build_feature_union():
    text_side = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            lowercase=True)),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
            lowercase=True)),
    ])
    return FeatureUnion([
        ("text", text_side),
        ("links", LinkFeatureExtractor()),
    ])


def build_candidates():
    """Three calibrated candidates, raw text in -> probability out."""
    c_lr = Pipeline([
        ("features", build_feature_union()),
        ("clf", CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=4000, random_state=SEED),
            cv=3, method="sigmoid")),
    ])
    c_nb = Pipeline([
        ("features", build_feature_union()),
        ("clf", CalibratedClassifierCV(
            estimator=MultinomialNB(alpha=0.1),
            cv=3, method="sigmoid")),
    ])
    c_svm = Pipeline([
        ("features", build_feature_union()),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(max_iter=4000, random_state=SEED),
            cv=3, method="sigmoid")),
    ])
    return {
        "logistic_regression": c_lr,
        "naive_bayes": c_nb,
        "linear_svm": c_svm,
    }


def main() -> None:
    print("=== STEP 4: full model (candidates + calibration) ===")
    clean = load_csv(PROCESSED_DIR / "clean_dataset.csv")
    assign = load_csv(PROCESSED_DIR / "split_assignments.csv")
    clean = clean.merge(assign[["row_id", "split"]], on="row_id")

    train = clean[clean["split"] == "train"].reset_index(drop=True)
    val = clean[clean["split"] == "validation"].reset_index(drop=True)
    y_train = train["label"].to_numpy()
    y_val = val["label"].to_numpy()

    candidates = build_candidates()
    comparison = []

    for name, pipe in candidates.items():
        print(f"fitting {name} ...")
        pipe.fit(train["text"].tolist(), y_train)
        proba = pipe.predict_proba(val["text"].tolist())[:, 1]
        pred = (proba >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum())
        pr_auc = float(average_precision_score(y_val, proba))
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        row = {
            "model": name,
            "validation_precision": round(precision, 4),
            "validation_recall": round(recall, 4),
            "validation_pr_auc": round(pr_auc, 4),
            "validation_fpr": round(fpr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        comparison.append(row)
        print("  ", row)

    # pick by PR-AUC, tie-break on lower FPR
    comparison.sort(key=lambda r: (-r["validation_pr_auc"], r["validation_fpr"]))
    best_name = comparison[0]["model"]
    best_pipe = candidates[best_name]
    print(f"\nselected: {best_name} "
          f"(PR-AUC {comparison[0]['validation_pr_auc']}, "
          f"FPR {comparison[0]['validation_fpr']})")

    # ---------- thresholds on validation (Step 6) ----------
    proba_val = best_pipe.predict_proba(val["text"].tolist())[:, 1]

    def precision_at(thr):
        pred = (proba_val >= thr).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        return tp / (tp + fp) if (tp + fp) else 0.0

    def recall_at_least(thr):
        # share of scams scoring strictly above thr (not flagged as low)
        scams = y_val == 1
        if not scams.any():
            return 1.0
        return float((proba_val[scams] > thr).mean())

    # high: smallest score threshold with validation precision >= 0.95
    grid = np.round(np.linspace(0.0, 1.0, 1001), 4)
    high_candidates = [t for t in grid if precision_at(t) >= 0.95]
    thr_high = float(min(high_candidates)) if high_candidates else 0.9
    # low: largest score threshold such that >= 99% of validation scams score above it
    low_candidates = [t for t in grid if recall_at_least(t) >= 0.99]
    thr_low = float(max(low_candidates)) if low_candidates else 0.2
    thr_low = min(thr_low, thr_high)  # low never above high

    print(f"thresholds: high={thr_high} low={thr_low} "
          f"(precision@high={precision_at(thr_high):.3f}, "
          f"scam-recall-above-low={recall_at_least(thr_low):.3f})")

    # ---------- save model + metadata ----------
    joblib.dump(best_pipe, ARTIFACTS_DIR / "scam_model.joblib")

    meta = {
        "chosen_model": best_name,
        "seed": SEED,
        "candidate_comparison": comparison,
        "thresholds": {
            "high_score": thr_high,   # 0-1 probability
            "low_score": thr_low,
            "high_precision_target": 0.95,
            "low_scams_recall_target": 0.99,
            "validation_precision_at_high": round(float(precision_at(thr_high)), 4),
            "validation_scam_recall_above_low": round(float(recall_at_least(thr_low)), 4),
        },
        "trained_on": "train split only (validation used for selection)",
        "link_features": [
            "has_link", "n_links", "is_shortener", "hyphen_host",
            "n_subdomains", "digit_in_host", "tld_risk",
        ],
    }
    (ARTIFACTS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("Wrote artifacts/scam_model.joblib and artifacts/model_meta.json")


if __name__ == "__main__":
    main()
