"""
step3_link_dependency.py - Step 3: link dependency test (the most important check).

Runs:
  Run A - normal:         baseline on ORIGINAL text, scored on ORIGINAL text.
  Run B - links removed   SAME model as A, scored on text with every link
                          replaced by [URL] (link_tools.strip_links).
                          (model was trained WITH links; only the TEST text changes)
  Run C - retrained       NEW baseline trained on stripped train/val/test/real
                          text, scored on stripped validation/test/real text.

For each run: precision, recall, F1, PR-AUC, false-positive rate on validation
and test, plus recall on the 94 real messages.

Conclusion rules (stated in plain words):
  - If Run B or Run C drops sharply from Run A (PR-AUC or real-message recall
    falls by >10 points), the model was leaning on the links.
  - If Run C stays close to Run A, the model learns from wording, not only links.
  - A drop is not automatically bad; scam links really do look different. A
    COLLAPSE is the red flag.
  - Shows 10 raw examples Run A got right and Run C got wrong.

Outputs:
  model/docs/link_dependency_test.md   (table + conclusion + examples)
  model/data/processed/link_dependency_meta.json   (machine-readable table)

Run from model/:  python src/step3_link_dependency.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SEED, PROCESSED_DIR, DOCS_DIR, load_csv, REAL_TEST,
                    build_baseline_pipeline, evaluate, language_label)
from link_tools import strip_links

DROP_ALERT = 0.10   # >10-point drop = "sharp" per the spec


def strip_series(texts) -> list[str]:
    return [strip_links(t) for t in texts]


def run_scores(model, texts, labels):
    proba = model.predict_proba(texts)[:, 1]
    return evaluate(labels, proba), proba


def fmt(r) -> str:
    """One-line rendering of a result dict (n/a where undefined)."""
    def v(k):
        x = r[k]
        return "n/a" if x is None else f"{x:.4f}"
    return (f"P={v('precision')} R={v('recall')} F1={v('f1')} "
            f"PR-AUC={v('pr_auc')} FPR={v('false_positive_rate')}")


def main() -> None:
    print("=== STEP 3: link dependency test ===")
    clean = load_csv(PROCESSED_DIR / "clean_dataset.csv")
    assign = load_csv(PROCESSED_DIR / "split_assignments.csv")
    clean = clean.merge(assign[["row_id", "split"]], on="row_id")
    real = load_csv(REAL_TEST)

    train = clean[clean["split"] == "train"].reset_index(drop=True)
    val = clean[clean["split"] == "validation"].reset_index(drop=True)
    test = clean[clean["split"] == "test"].reset_index(drop=True)

    stripped = lambda df: df.assign(text_stripped=strip_series(df["text"]))
    train_s, val_s, test_s, real_s = stripped(train), stripped(val), stripped(test), stripped(real)

    results = {}   # run -> dict of result dicts
    models = {}

    # ---------------- Run A ----------------
    print("Run A: baseline on original text")
    mA = build_baseline_pipeline()
    mA.fit(train["text"], train["label"])
    models["A"] = mA
    results["A"] = {
        "validation": run_scores(mA, val["text"].tolist(), val["label"].tolist())[0],
        "test": run_scores(mA, test["text"].tolist(), test["label"].tolist())[0],
        "real": run_scores(mA, real["text"].tolist(), real["label"].tolist())[0],
    }

    # ---------------- Run B ----------------
    print("Run B: model A, scored on stripped text")
    results["B"] = {
        "validation": run_scores(mA, val_s["text_stripped"].tolist(), val["label"].tolist())[0],
        "test": run_scores(mA, test_s["text_stripped"].tolist(), test["label"].tolist())[0],
        "real": run_scores(mA, real_s["text_stripped"].tolist(), real["label"].tolist())[0],
    }

    # ---------------- Run C ----------------
    print("Run C: fresh baseline trained on stripped text")
    mC = build_baseline_pipeline()
    mC.fit(train_s["text_stripped"], train_s["label"])
    models["C"] = mC
    results["C"] = {
        "validation": run_scores(mC, val_s["text_stripped"].tolist(), val["label"].tolist())[0],
        "test": run_scores(mC, test_s["text_stripped"].tolist(), test["label"].tolist())[0],
        "real": run_scores(mC, real_s["text_stripped"].tolist(), real["label"].tolist())[0],
    }

    # ---------------- examples: A right, C wrong ----------------
    proba_C_test = models["C"].predict_proba(test_s["text_stripped"].tolist())[:, 1]
    proba_A_test = models["A"].predict_proba(test["text"].tolist())[:, 1]
    y_test = test["label"].tolist()
    pred_C = (proba_C_test >= 0.5).astype(int)
    pred_A = (proba_A_test >= 0.5).astype(int)

    mask = (pred_A == y_test) & (pred_C != y_test)   # A right, C wrong
    examples = []
    for i in np.flatnonzero(mask)[:10]:
        examples.append({
            "text": test["text"].iloc[i],
            "label": int(y_test[i]),
            "stripped": test_s["text_stripped"].iloc[i],
            "prob_A": round(float(proba_A_test[i]), 4),
            "prob_C": round(float(proba_C_test[i]), 4),
            "pred_A": int(pred_A[i]),
            "pred_C": int(pred_C[i]),
            "links_found": __import__("link_tools").extract_links(test["text"].iloc[i]),
        })

    # ---------------- conclusion ----------------
    A_test, B_test, C_test = results["A"]["test"], results["B"]["test"], results["C"]["test"]
    A_real, B_real, C_real = results["A"]["real"], results["B"]["real"], results["C"]["real"]

    def drop(a, b):   # A -> B drop in points
        return (a["pr_auc"] - b["pr_auc"]) if (a["pr_auc"] is not None and b["pr_auc"] is not None) else None

    b_pr_drop = drop(A_test, B_test)
    c_pr_drop = drop(A_test, C_test)
    b_real_drop = (A_real["recall"] - B_real["recall"]) if A_real["recall"] is not None else None
    c_real_drop = (A_real["recall"] - C_real["recall"]) if A_real["recall"] is not None else None

    sharp = []
    if b_pr_drop is not None and b_pr_drop > DROP_ALERT:
        sharp.append(f"Run B PR-AUC fell {b_pr_drop:.2f} points below Run A")
    if c_pr_drop is not None and c_pr_drop > DROP_ALERT:
        sharp.append(f"Run C PR-AUC fell {c_pr_drop:.2f} points below Run A")
    if b_real_drop is not None and b_real_drop > DROP_ALERT:
        sharp.append(f"Run B real-message recall fell {b_real_drop:.2f} below Run A")
    if c_real_drop is not None and c_real_drop > DROP_ALERT:
        sharp.append(f"Run C real-message recall fell {c_real_drop:.2f} below Run A")

    conclusion = []
    if sharp:
        conclusion.append("LINK DEPENDENCE DETECTED: the model leans on links. "
                          "The drops show it: " + "; ".join(sharp) + ". "
                          "This is not automatically wrong - scam links really do "
                          "look different from legitimate ones - but it means the "
                          "score would collapse on scam-free wording, which is a "
                          "red flag for deployment.")
    else:
        conclusion.append("NO COLLAPSE: Run C stays close to Run A "
                          f"(PR-AUC change {c_pr_drop:.2f} points if defined; "
                          f"real recall change {c_real_drop:.2f}). "
                          "The model learns from wording, not only from links.")

    # ---------------- save markdown ----------------
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Link dependency test (Step 3)\n")
    lines.append("Question: does the model detect scams from the WORDING, or does it "
                 "just spot scammy links? Three runs compare.\n")
    lines.append("**Run A** - normal: baseline trained and scored on the original text.\n")
    lines.append("**Run B** - links removed at test time: the SAME model as Run A, "
                 "scored on text where every link was replaced by [URL].\n")
    lines.append("**Run C** - retrained without links: a NEW baseline trained on "
                 "link-stripped train/validation/test text and scored on stripped text.\n")
    lines.append("All models are the Step-2 baseline: word TF-IDF (1-2 grams) + "
                 "character TF-IDF (3-5 grams) + logistic regression.\n")

    lines.append("## Results table\n")
    lines.append("| run | set | precision | recall | f1 | PR-AUC | false-positive rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for run in ("A", "B", "C"):
        for nm in ("validation", "test"):
            r = results[run][nm]
            lines.append(f"| {run} | {nm} | {r['precision']} | {r['recall']} | "
                         f"{r['f1']} | {r['pr_auc']} | {r['false_positive_rate']} |")
        r = results[run]["real"]
        lines.append(f"| {run} | real-94 (recall only) | n/a | {r['recall']} | n/a | n/a | n/a |")
    lines.append("")

    lines.append("## Conclusion\n")
    for c in conclusion:
        lines.append(f"- {c}")
    lines.append("")
    lines.append(f"- Run A -> real recall {A_real['recall']:.3f}; "
                 f"Run B {B_real['recall']:.3f}; Run C {C_real['recall']:.3f}.")
    lines.append(f"- Run A -> test PR-AUC {A_test['pr_auc']}; Run B {B_test['pr_auc']}; "
                 f"Run C {C_test['pr_auc']}.")
    lines.append("")
    lines.append("> A drop is not automatically bad: scam links really do look different "
                 "> from legitimate ones. A COLLAPSE (>10 points) is the red flag.\n")

    lines.append("## 10 examples: Run A got right, Run C got wrong\n")
    lines.append("These are raw test messages where the normal model (A) predicted "
                 "correctly but the no-links model (C) erred. They show exactly which "
                 "messages depend on link signals.\n")
    for i, ex in enumerate(examples, 1):
        lbl = "scam" if ex["label"] == 1 else "benign"
        lines.append(f"### Example {i} (truth: {lbl})\n")
        lines.append(f"- Original: {ex['text']}")
        if ex["links_found"]:
            lines.append(f"- Links in original: {ex['links_found']}")
        lines.append(f"- Stripped: {ex['stripped']}")
        lines.append(f"- Run A: probability {ex['prob_A']} -> correct")
        lines.append(f"- Run C: probability {ex['prob_C']} -> wrong")
        lines.append("")

    DOCS_DIR.joinpath("link_dependency_test.md").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "runs": results,
        "examples": examples,
        "conclusion": conclusion,
        "drops": {"run_B_pr_drop": b_pr_drop, "run_C_pr_drop": c_pr_drop,
                  "run_B_real_drop": b_real_drop, "run_C_real_drop": c_real_drop},
    }
    (PROCESSED_DIR / "link_dependency_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nRun A :", results["A"])
    print("Run B :", results["B"])
    print("Run C :", results["C"])
    print("Conclusion:", conclusion[0])
    print(f"Examples A-right/C-wrong: {len(examples)}")
    print("Wrote docs/link_dependency_test.md")


if __name__ == "__main__":
    main()
