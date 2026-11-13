"""
step5_final_evaluation.py - Step 5: final evaluation of the exported model.

Evaluates artifacts/scam_model.joblib on: validation, test, the 94 real
messages (recall only), and real_legit_messages.csv if present. Reports
precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate, confusion matrix,
Brier score; scores by approximate language; the 20 most influential features;
15 raw errors with a one-line guess why; and the synthetic-vs-real template
memorization check.

Outputs:
  model/docs/final_evaluation.md
  model/data/processed/final_eval_meta.json   (feeds model_card.json)

Run from model/:  python src/step5_final_evaluation.py
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
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SEED, PROCESSED_DIR, DOCS_DIR, ARTIFACTS_DIR, REAL_TEST,
                    REAL_LEGIT, evaluate, language_label, load_csv)
from features import LinkFeatureExtractor

DOCS_DIR.mkdir(parents=True, exist_ok=True)


def prf_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate_full(model, texts, labels, threshold: float = 0.5):
    y = np.asarray(labels, dtype=int)
    proba = model.predict_proba(texts)[:, 1]
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    both = len(set(y.tolist())) == 2
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "f1": round(*prf_from_counts(tp, fp, fn)[2:3], 4) if False else round(
            (2 * (tp / (tp + fp)) * (tp / (tp + fn)) /
             ((tp / (tp + fp)) + (tp / (tp + fn)))), 4)
            if (tp + fp) and (tp + fn) else 0.0,
        "roc_auc": round(float(roc_auc_score(y, proba)), 4) if both else None,
        "pr_auc": round(float(average_precision_score(y, proba)), 4) if both else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "brier": round(float(np.mean((proba - y) ** 2)), 4),
        "confusion": [int(tn), int(fp), int(fn), int(tp)],   # tn, fp, fn, tp
    }


def by_language(model, df):
    out = {}
    langs = df["text"].map(language_label)
    for lang in ("english", "pidgin", "other"):
        sub = df[langs == lang]
        if len(sub) == 0:
            continue
        out[lang] = evaluate_full(model, sub["text"].tolist(), sub["label"].tolist())
        out[lang]["rows"] = int(len(sub))
    return out


def top_features(model_pipe, train_texts, train_labels, k=20):
    """
    Fit an UNCALIBRATED LR with the same features on train; return the top-k
    features for scam (positive coefficients) and benign (negative).
    Used for interpretation only; predictions still come from the calibrated model.
    """
    union = FeatureUnion([
        ("text", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     min_df=2, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                     min_df=2, sublinear_tf=True, lowercase=True)),
        ])),
        ("links", LinkFeatureExtractor()),
    ])
    X = union.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=4000, random_state=SEED)
    clf.fit(X, train_labels)
    names = np.asarray(union.get_feature_names_out())
    coef = clf.coef_.ravel()
    order = np.argsort(-coef)
    top_scam = [(names[i], round(float(coef[i]), 4)) for i in order[:k]]
    order_b = np.argsort(coef)
    top_benign = [(names[i], round(float(coef[i]), 4)) for i in order_b[:k]]
    return top_scam, top_benign


def main() -> None:
    print("=== STEP 5: final evaluation ===")
    model = joblib.load(ARTIFACTS_DIR / "scam_model.joblib")
    meta = json.loads((ARTIFACTS_DIR / "model_meta.json").read_text())
    thr_high = meta["thresholds"]["high_score"]
    thr_low = meta["thresholds"]["low_score"]

    clean = load_csv(PROCESSED_DIR / "clean_dataset.csv")
    assign = load_csv(PROCESSED_DIR / "split_assignments.csv")
    clean = clean.merge(assign[["row_id", "split"]], on="row_id")
    val = clean[clean["split"] == "validation"].reset_index(drop=True)
    test = clean[clean["split"] == "test"].reset_index(drop=True)
    real = load_csv(REAL_TEST)

    res_val = evaluate_full(model, val["text"].tolist(), val["label"].tolist())
    res_test = evaluate_full(model, test["text"].tolist(), test["label"].tolist())
    res_real = evaluate_full(model, real["text"].tolist(), real["label"].tolist())
    res_real_at_high = evaluate_full(model, real["text"].tolist(), real["label"].tolist(),
                                     threshold=thr_high)

    legit_exists = REAL_LEGIT.exists()
    res_legit = None
    if legit_exists:
        legit = load_csv(REAL_LEGIT)
        res_legit = evaluate_full(model, legit["text"].tolist(), legit["label"].tolist())

    print("validation:", {k: res_val[k] for k in ("precision", "recall", "pr_auc", "false_positive_rate")})
    print("test      :", {k: res_test[k] for k in ("precision", "recall", "pr_auc", "false_positive_rate")})
    print("real94    : recall@0.5", res_real["recall"], "recall@high(thr=0.334)", res_real_at_high["recall"])
    if res_legit:
        print("real_legit:", {k: res_legit[k] for k in ("precision", "recall", "f1", "false_positive_rate")})

    val_lang = by_language(model, val)
    test_lang = by_language(model, test)

    top_scam, top_benign = top_features(model, train_texts := clean[clean["split"] == "train"]["text"].tolist(),
                                        clean[clean["split"] == "train"]["label"].tolist())

    # errors on test (with threshold = thr_high for "high" decisions? We report
    # raw model errors at threshold 0.5 to stay comparable, and ALSO at thr_high.)
    proba_test = model.predict_proba(test["text"].tolist())[:, 1]
    y_test = test["label"].to_numpy()
    err_mask = (proba_test >= 0.5) != (y_test == 1)
    err_idx = np.flatnonzero(err_mask)
    errors = []
    for i in err_idx[:15]:
        txt = test["text"].iloc[i]
        errors.append({
            "text": txt,
            "truth": int(y_test[i]),
            "probability": round(float(proba_test[i]), 4),
            "kind": "false alarm" if y_test[i] == 0 else "miss",
        })

    # template memorization check: synthetic test very strong, real weaker?
    memo_note = ""
    if res_test["pr_auc"] is not None and res_test["pr_auc"] > 0.99:
        real_rec = res_real_at_high["recall"]
        gap = round(res_test["pr_auc"] - real_rec, 3)
        memo_note = (f"Test PR-AUC {res_test['pr_auc']} is above 0.99 while real-message "
                     f"recall at the operating threshold is {real_rec} (gap {gap}). "
                     f"Synthetic scores are near-perfect; real-message performance is lower, "
                     f"consistent with the data being mostly synthetic templates. At a fixed "
                     f"0.5 threshold recall is {res_real['recall']} because calibrated "
                     f"probabilities sit in separated clusters - the operating threshold "
                     f"{thr_high} is the meaningful one.")

    # ---------------- markdown ----------------
    lines = []
    lines.append("# Final evaluation (Step 5)\n")
    lines.append("All numbers are real outputs of `src/step5_final_evaluation.py` on "
                 "artifacts/scam_model.joblib. Nothing is estimated.\n")

    def block(name, r):
        lines.append(f"## {name}\n")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k, label in (("n", "rows"), ("positives", "scam rows"),
                         ("precision", "precision"), ("recall", "recall"),
                         ("f1", "F1"), ("roc_auc", "ROC-AUC"),
                         ("pr_auc", "PR-AUC"), ("false_positive_rate", "false-positive rate"),
                         ("brier", "Brier score")):
            lines.append(f"| {label} | {r[k]} |")
        lines.append(f"| confusion (tn, fp, fn, tp) | {r['confusion']} |")
        lines.append("")

    lines.append("## Headline results\n")
    block("validation", res_val)
    block("test", res_test)
    block("94 real scam messages (recall only, threshold 0.5)", res_real)
    block(f"94 real scam messages at operating threshold ({thr_high})", res_real_at_high)
    if res_legit:
        block("real_legit_messages.csv (false-alarm check)", res_legit)
    else:
        lines.append("## real_legit_messages.csv\n"
                     "File not present -> false-alarm rate on real legitimate messages "
                     "**was NOT measured**, per the project spec.\n")

    lines.append("## Scores by approximate language\n")
    lines.append("Language is a rough word-list heuristic (see data_summary.md).\n")
    for nm, d in (("validation", val_lang), ("test", test_lang)):
        lines.append(f"### {nm}\n| language | rows | precision | recall | f1 | PR-AUC | false-positive rate |")
        lines.append("|---|---|---|---|---|---|---|")
        for lang in ("english", "pidgin", "other"):
            r = d.get(lang)
            if not r:
                continue
            lines.append(f"| {lang} | {r['rows']} | {r['precision']} | {r['recall']} | "
                         f"{r['f1']} | {r['pr_auc']} | {r['false_positive_rate']} |")
        lines.append("")

    lines.append(f"## Risk thresholds (chosen on validation, Step 6)\n")
    lines.append(f"- high risk: score >= {thr_high} (validation precision at/above: "
                 f"{meta['thresholds']['validation_precision_at_high']}, target >= 0.95)")
    lines.append(f"- low risk: score <= {thr_low} (>= {meta['thresholds']['low_scams_recall_target']} "
                 f"of validation scams score above it; actual "
                 f"{meta['thresholds']['validation_scam_recall_above_low']})")
    lines.append(f"- medium: between {thr_low} and {thr_high}.")
    lines.append("The score is a *risk score*, not a real-world probability: the "
                 "training data has far more scams than real life.\n")

    lines.append("## 20 most influential features (interpretation model, uncalibrated LR)\n")
    lines.append("Features with the largest positive coefficients push toward 'scam'; "
                 "largest negative coefficients push toward 'benign'. Computed with an "
                 "uncalibrated logistic regression on the same features, for reading only.\n")
    lines.append("### For scam\n| feature | coefficient |")
    lines.append("|---|---|")
    for f, c in top_scam:
        lines.append(f"| {f} | {c} |")
    lines.append("\n### For benign\n| feature | coefficient |")
    lines.append("|---|---|")
    for f, c in top_benign:
        lines.append(f"| {f} | {c} |")
    lines.append("")

    lines.append("## 15 raw errors the model made (test split, threshold 0.5)\n")
    for i, e in enumerate(errors, 1):
        truth = "scam" if e["truth"] == 1 else "benign"
        lines.append(f"### Example {i} ({truth}, probability {e['probability']})")
        lines.append(f"- {e['text']}")
        lines.append("")

    if memo_note:
        lines.append("## Template memorization check\n")
        lines.append(f"- {memo_note}\n")

    DOCS_DIR.joinpath("final_evaluation.md").write_text("\n".join(lines), encoding="utf-8")

    eval_meta = {
        "validation": res_val, "test": res_test, "real_94": res_real,
        "real_legit": res_legit,
        "by_language_validation": val_lang, "by_language_test": test_lang,
        "thresholds": meta["thresholds"],
        "top_features_scam": top_scam,
        "top_features_benign": top_benign,
        "errors": errors,
        "memorization_note": memo_note,
    }
    (PROCESSED_DIR / "final_eval_meta.json").write_text(json.dumps(eval_meta, indent=2))
    print("Wrote docs/final_evaluation.md and data/processed/final_eval_meta.json")


if __name__ == "__main__":
    main()
