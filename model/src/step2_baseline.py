"""
step2_baseline.py - Step 2: the plain baseline (run BEFORE anything fancy).

Model spec (fixed here so Step 3 can re-use EXACTLY the same spec):
  word TF-IDF (1-2 grams) + character TF-IDF (3-5 grams, lowercase),
  combined with FeatureUnion,
  logistic regression classifier,
  trained on the `text` of the train split. No hand-made link features.

Scoring: precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate,
Brier score + confusion counts, on validation, test and the 94 real messages
(recall only there).

Saves:
  model/docs/baseline_results.md      (plain summary with all numbers)
  model/data/processed/baseline_meta.json   (machine-readable copy)

Run from model/:  python src/step2_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SEED, PROCESSED_DIR, DOCS_DIR, load_csv, REAL_TEST,
                    evaluate, language_label)

# ---------------------------------------------------------------------------
# Baseline model spec - shared with Step 3 via common.build_baseline_pipeline
# ---------------------------------------------------------------------------
def build_baseline_pipeline():
    """Word TF-IDF + char TF-IDF + logistic regression (Step 2 spec)."""
    union = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            lowercase=True)),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
            lowercase=True)),
    ])
    return Pipeline([
        ("features", union),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def scores_for(model, texts, labels) -> dict:
    """predict_proba + evaluate; handles single-class data gracefully."""
    proba = model.predict_proba(texts)[:, 1]
    return evaluate(labels, proba)


def confusion_text(y_true, proba, threshold: float = 0.5) -> str:
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return (f"tn={tn} fp={fp} fn={fn} tp={tp}")


def main() -> None:
    print("=== STEP 2: baseline ===")
    clean = load_csv(PROCESSED_DIR / "clean_dataset.csv")
    assign = load_csv(PROCESSED_DIR / "split_assignments.csv")
    clean = clean.merge(assign[["row_id", "split"]], on="row_id")

    train = clean[clean["split"] == "train"].reset_index(drop=True)
    val   = clean[clean["split"] == "validation"].reset_index(drop=True)
    test  = clean[clean["split"] == "test"].reset_index(drop=True)
    real  = load_csv(REAL_TEST)

    print(f"train={len(train)} val={len(val)} test={len(test)} real={len(real)}")

    # ---------------- train baseline ----------------
    model = build_baseline_pipeline()
    model.fit(train["text"], train["label"])

    # ---------------- score ----------------
    res_val  = scores_for(model, val["text"].tolist(),  val["label"].tolist())
    res_test = scores_for(model, test["text"].tolist(), test["label"].tolist())
    res_real = scores_for(model, real["text"].tolist(), real["label"].tolist())

    print("\nvalidation :", res_val)
    print("test       :", res_test)
    print("real (94)  :", res_real)

    # ---------------- confusion matrices ----------------
    proba_val  = model.predict_proba(val["text"].tolist())[:, 1]
    proba_test = model.predict_proba(test["text"].tolist())[:, 1]
    proba_real = model.predict_proba(real["text"].tolist())[:, 1]

    res_val["confusion"] = confusion_text(val["label"].tolist(), proba_val)
    res_test["confusion"] = confusion_text(test["label"].tolist(), proba_test)
    res_real["confusion"] = confusion_text(real["label"].tolist(), proba_real)

    val_lang = val.assign(lang=val["text"].map(language_label))
    test_lang = test.assign(lang=test["text"].map(language_label))
    lang_df_map = {"validation": val_lang, "test": test_lang}

    # ---------------- by approximate language ----------------
    lang_results = {}
    for name, lang_df in (("validation", val_lang), ("test", test_lang)):
        lang_results[name] = {}
        for lang in ("english", "pidgin", "other"):
            sub = lang_df[lang_df["lang"] == lang]
            if len(sub) == 0:
                continue
            lang_results[name][lang] = scores_for(
                model, sub["text"].tolist(), sub["label"].tolist())
    print("by language:", {k: {l: (round(v["pr_auc"], 3)
                                   if v["pr_auc"] is not None else None)
                             for l, v in d.items()}
                           for k, d in lang_results.items()})

    # ---------------- save baseline_results.md ----------------
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Baseline results (Step 2)\n")
    lines.append("Plain model: word TF-IDF (1-2 grams) + character TF-IDF (3-5 grams) + "
                 "logistic regression. Trained on `text` only - no link features. "
                 "All numbers are real outputs of this run, not estimates.\n")

    def show(name, r):
        lines.append(f"## {name}\n")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k in ("n", "positives", "tp", "fp", "fn", "tn", "precision",
                  "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "brier"):
            lines.append(f"| {k} | {r[k]} |")
        lines.append(f"| confusion | {r['confusion']} |")
        lines.append("")

    lines.append("## Test-set comparison\n")
    lines.append("All three runs use the SAME trained model.\n")
    show("validation (4,447 rows)", res_val)
    show("test (4,448 rows)", res_test)
    show("94 real scam messages (recall only)", res_real)

    lines.append("## By approximate language\n")
    lines.append("Language is a rough word-list heuristic (see data_summary.md); "
                 "'other' includes French/Spanish/Portuguese and non-Latin script.\n")
    for nm in ("validation", "test"):
        lines.append(f"### {nm}\n| language | precision | recall | f1 | pr_auc | false_positive_rate | rows |")
        lines.append("|---|---|---|---|---|---|---|")
        for lang in ("english", "pidgin", "other"):
            r = lang_results.get(nm, {}).get(lang)
            if not r:
                continue
            rows_n = int((lang_df_map[nm]["lang"] == lang).sum())
            lines.append(f"| {lang} | {r['precision']} | {r['recall']} | "
                         f"{r['f1']} | {r['pr_auc']} | {r['false_positive_rate']} | {rows_n} |")
        lines.append("")

    lines.append("## Reading the numbers\n")
    lines.append("- PR-AUC rewards ranking scams above benign messages; it is the "
                 "headline number because the data is scam-heavy (68%).")
    lines.append("- False-positive rate = share of benign messages wrongly flagged.")
    lines.append("- On the 94 real messages we can measure RECALL only (they are all "
                 "scams), so no false-alarm rate is available for real messages yet.")
    DOCS_DIR.joinpath("baseline_results.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------- machine-readable copy ----------------
    meta = {
        "model": "baseline: word TF-IDF (1-2) + char TF-IDF (3-5) + logistic regression",
        "seed": SEED,
        "validation": res_val,
        "test": res_test,
        "real_94_recall_only": res_real,
        "by_language": lang_results,
    }
    (PROCESSED_DIR / "baseline_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nWrote docs/baseline_results.md and data/processed/baseline_meta.json")


if __name__ == "__main__":
    main()
