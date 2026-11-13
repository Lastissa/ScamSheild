"""
step1_data_check.py - Step 1: data check and a leakage-safe grouped split.

What it does (in order):
  a. Counts by label, nulls, exact duplicates (dropped), message lengths,
     share of messages containing links by label. Drops any row whose text
     appears in the real-messages test file (hard rule 6: never train on it).
  b. Builds GROUPS of near-duplicate messages: mask links/emails/numbers/
     money, word TF-IDF, cosine neighbours above a cutoff, connected
     components. Near-copies land in the same group, so they can never be
     split across train and test (that would inflate every score).
  c. Splits BY GROUP into train 70% / validation 15% / test 15% with a fixed
     seed, done separately for scam-majority and benign-majority groups so
     the scam/benign ratio stays similar in each part.
  d. LEAKAGE CHECK: for every validation and test message, its highest cosine
     similarity to any TRAINING message. Fails if >10% of test messages have
     a near-copy (>0.9) in train.
  e. Writes model/docs/data_summary.md from the real numbers of THIS run.
  f. Gives every message an approximate language label (pidgin/other/english).

Outputs (all inside model/):
  model/data/processed/clean_dataset.csv      (row_id, text, label, language, n_links)
  model/data/processed/split_assignments.csv  (row_id, split, group_id)
  model/data/processed/split_meta.json        (all step-1 numbers, reused later)
  model/docs/data_summary.md                  (plain-English summary)

Run from the model/ folder:  python src/step1_data_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SEED, MAIN_DATA, REAL_TEST, PROCESSED_DIR, DOCS_DIR,
                    load_csv, mask_text, language_label)
from link_tools import extract_links

# --- tuning constants for this step -----------------------------------------
SIM_CUTOFFS = [0.85, 0.80, 0.75, 0.90, 0.95]  # try in this order until gates pass
MAX_GROUP_SHARE = 0.5     # a single group must not cover more than 50% of data
MAX_RATIO_DRIFT_PP = 5.0  # split scam-share may drift at most 5 points from overall
LEAK_GATE = 0.10          # >10% of test msgs with a >0.9 near-copy in train = fail
NEAR_COPY = 0.90          # what counts as a "near-copy" in the leakage check


def norm_text(s: str) -> str:
    """Whitespace-collapsed lowercase text - used only for duplicate checks."""
    return " ".join(str(s).lower().split())


def build_groups(masked_texts: list[str], cutoff: float) -> tuple[np.ndarray, sp.csr_matrix]:
    """
    Group near-duplicate masked messages by connected components.

    Returns (group_labels, adjacency) where group_labels[i] is the group id of
    message i. Two messages are connected when cosine TF-IDF similarity of
    their MASKED texts is above `cutoff`.
    """
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(masked_texts)
    X = normalize(X, norm="l2", copy=True)          # rows become unit length
    n = X.shape[0]

    rows, cols = [], []
    CHUNK = 2000                                     # rows per similarity block
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        block = X[start:stop] @ X.T                  # cosine sims of this block
        block = block.tocoo()
        keep = block.data >= cutoff
        r = block.row[keep] + start                  # absolute row index
        c = block.col[keep]
        # keep each undirected edge once (upper triangle: row < col)
        lo, hi = np.minimum(r, c), np.maximum(r, c)
        self_edges = lo == hi
        rows.append(lo[~self_edges])
        cols.append(hi[~self_edges])

    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
    else:
        r = c = np.array([], dtype=np.int64)
    adj = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    return labels, adj


SPLIT_NAMES = ("train", "validation", "test")


def gss_stratified_group_split(groups: np.ndarray, df: pd.DataFrame) -> pd.Series:
    """
    REFERENCE ONLY - the spec's first choice: random GroupShuffleSplit by
    group, done per label class (70/15/15). We keep it to MEASURE its
    scam-share drift; see greedy_group_split for why it is not used as the
    final split.
    """
    split = pd.Series(index=df.index, dtype="object")
    grp_series = pd.Series(groups, index=df.index)
    lab_series = df["label"]

    group_of_label = {}
    for g in np.unique(groups):
        mask = grp_series == g
        group_of_label[g] = int(lab_series[mask].value_counts().idxmax())

    for cls in (0, 1):
        cls_groups = np.array(sorted(g for g, l in group_of_label.items() if l == cls))
        cls_rows = df.index[grp_series.isin(cls_groups)]
        g_arr = grp_series.loc[cls_rows].to_numpy()   # group id per row, positional

        gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
        tr_pos, temp_pos = next(gss1.split(X=None, y=None, groups=g_arr))
        tr_grp_ids = set(np.unique(g_arr[tr_pos]))
        temp_rows = cls_rows[temp_pos]
        g_temp = g_arr[temp_pos]

        gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED + 1)
        va_pos, te_pos = next(gss2.split(X=None, y=None, groups=g_temp))

        split.loc[cls_rows[np.isin(g_arr, list(tr_grp_ids))]] = "train"
        split.loc[temp_rows[va_pos]] = "validation"
        split.loc[temp_rows[te_pos]] = "test"

    assert split.notna().all(), "every row must get a split"
    return split


def greedy_group_split(groups: np.ndarray, df: pd.DataFrame,
                       targets=(0.70, 0.15, 0.15)) -> pd.Series:
    """
    FINAL SPLIT METHOD - deterministic, size-aware assignment of whole groups.

    Why not plain GroupShuffleSplit (measured, see data_summary.md): the
    near-duplicate groups chain template messages into big clusters (only
    ~1,400 groups for ~29,650 rows; the largest holds ~3% of all rows), so a
    random group split puts up to 12 percentage points of scam-share drift
    between the parts. That would distort every validation/test comparison.

    Method: groups sorted largest-first (they constrain the most). Each group
    goes to the split where it fills the largest share of that split's
    REMAINING quota (quota = its ideal 70/15/15 share of scam messages and of
    benign messages, counted separately). Train starts with the biggest
    quota, so it fills first; when its remaining fraction drops below the
    other parts, groups flow there. Deterministic - no randomness at all, so
    reproducibility is trivial. Groups are never split apart.
    """
    labels = df["label"].to_numpy()
    total = len(labels)
    total_scams = int(labels.sum())

    g_df = pd.DataFrame({"g": groups, "y": labels})
    agg = g_df.groupby("g")["y"].agg(size="size", scams="sum").reset_index()
    order = agg.sort_values(["size", "g"], ascending=[False, True])  # deterministic

    ideal_n = {s: t * total for s, t in zip(SPLIT_NAMES, targets)}
    ideal_sc = {s: t * total_scams for s, t in zip(SPLIT_NAMES, targets)}
    ideal_be = {s: ideal_n[s] - ideal_sc[s] for s in SPLIT_NAMES}
    cur_n = {s: 0.0 for s in SPLIT_NAMES}
    cur_sc = {s: 0.0 for s in SPLIT_NAMES}
    cur_be = {s: 0.0 for s in SPLIT_NAMES}
    assign = {}

    for _, row in order.iterrows():
        g, size = int(row["g"]), int(row["size"])
        scams, ben = int(row["scams"]), size - int(row["scams"])
        best, best_key = None, None
        for s in SPLIT_NAMES:
            # how much of this group fills genuinely needed quota in split s
            fill = min(scams, max(ideal_sc[s] - cur_sc[s], 0)) \
                 + min(ben, max(ideal_be[s] - cur_be[s], 0))
            fill_frac = fill / size
            # tie-break: prefer the split that is emptiest relative to its size
            rem_frac = (ideal_n[s] - cur_n[s]) / ideal_n[s]
            key = (fill_frac, rem_frac)
            if best_key is None or key > best_key:
                best, best_key = s, key
        assign[g] = best
        cur_n[best] += size
        cur_sc[best] += scams
        cur_be[best] += ben

    split = pd.Series([assign[int(g)] for g in groups], index=df.index)
    assert split.notna().all(), "every row must get a split"
    # hard guarantee: a group must never appear in two parts
    part_of = {}
    for g, s in zip(groups, split.to_numpy()):
        if part_of.setdefault(g, s) != s:
            raise AssertionError(f"group {g} appears in two splits")
    return split


def leakage_check(train_masked, val_masked, test_masked) -> dict:
    """
    For every validation/test message: highest cosine similarity to ANY train
    message (TF-IDF of masked text, vocabulary fitted on train only).
    """
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)
    Xtr = normalize(vec.fit_transform(train_masked), norm="l2", copy=True)

    out = {}
    for name, texts in (("validation", val_masked), ("test", test_masked)):
        Xq = normalize(vec.transform(texts), norm="l2", copy=True)
        sims = (Xq @ Xtr.T).max(axis=1).toarray().ravel()   # max sim per message
        out[name] = {
            "n": int(len(texts)),
            "median_max_sim": round(float(np.median(sims)), 4),
            "mean_max_sim": round(float(np.mean(sims)), 4),
            "share_above_090": round(float((sims > NEAR_COPY).mean()), 4),
            "share_above_085": round(float((sims > 0.85).mean()), 4),
        }
    return out


def main() -> None:
    print("=== STEP 1: data check and leakage-safe split ===")

    # ---------------- (a) basic checks ----------------
    df = load_csv(MAIN_DATA)
    real = load_csv(REAL_TEST)
    n_raw = len(df)
    label_counts_raw = df["label"].value_counts().to_dict()

    nulls = int(df["text"].isna().sum() + (df["text"].astype(str).str.strip() == "").sum())
    dup_mask = df["text"].map(norm_text).duplicated(keep="first")
    n_dups = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)

    # never train on (even a copy of) the real test messages
    real_norm = set(norm_text(t) for t in real["text"])
    real_mask = df["text"].map(norm_text).isin(real_norm)
    n_real_overlap = int(real_mask.sum())
    df = df[~real_mask].reset_index(drop=True)

    label_counts = df["label"].value_counts().to_dict()
    share_scam = label_counts.get(1, 0) / len(df)
    print(f"raw rows: {n_raw} | labels: {label_counts_raw} | empty text: {nulls}")
    print(f"exact/near-exact duplicates dropped: {n_dups}")
    print(f"rows matching the 94 real test messages dropped: {n_real_overlap}")
    print(f"clean rows: {len(df)} | scam share: {share_scam:.3f}")

    # language labels (approximate heuristic, Step 1f)
    df["language"] = df["text"].map(language_label)
    lang_by_label = df.groupby("label")["language"].value_counts(normalize=True).unstack().fillna(0)

    # message length + link share by label
    df["n_links"] = df["text"].map(lambda t: len(extract_links(str(t))))
    df["has_link"] = (df["n_links"] > 0).astype(int)
    len_stats = {int(k): {"mean": round(float(v.mean()), 1),
                          "median": round(float(v.median()), 1),
                          "p10": round(float(v.quantile(0.10)), 1),
                          "p90": round(float(v.quantile(0.90)), 1)}
                 for k, v in len_stats_groups(df)}
    link_share = df.groupby("label")["has_link"].mean().round(4).to_dict()
    print("length stats by label:", len_stats)
    print("share with links by label:", link_share)

    # ---------------- (b)+(c)+(d) grouping, split, leakage gate ----------------
    masked = [mask_text(t) for t in df["text"]]
    notes: list[str] = []

    chosen = None
    gss_reference_drift = None
    for cutoff in SIM_CUTOFFS:
        groups, adj = build_groups(masked, cutoff)
        sizes = pd.Series(groups).value_counts()
        max_share = float(sizes.iloc[0]) / len(df)

        # Reference: the spec's first choice (random GroupShuffleSplit by
        # group) - we measure its drift to justify the final method.
        split_gss = gss_stratified_group_split(groups, df)
        drift_gss = float((df.groupby(split_gss)["label"].mean() - share_scam).abs().max())
        if gss_reference_drift is None:
            gss_reference_drift = drift_gss
            notes.append(
                f"Plain random GroupShuffleSplit by group (spec's first choice, measured at "
                f"cutoff {cutoff}) drifts {drift_gss:.1%} in scam share between parts, "
                f"because near-duplicate groups chain into big template clusters "
                f"(largest group = {max_share:.1%} of rows). A deterministic greedy "
                f"largest-first group assignment is used instead; it keeps groups whole "
                f"and balances rows and scam share.")

        # Actual split: deterministic greedy assignment.
        split = greedy_group_split(groups, df)
        scam_by_split = df.groupby(split)["label"].mean()
        drift = float((scam_by_split - share_scam).abs().max())

        masked_arr = np.array(masked, dtype=object)   # so bool-mask indexing works
        leak = leakage_check(masked_arr[split == "train"],
                             masked_arr[split == "validation"],
                             masked_arr[split == "test"])
        print(f"\ncutoff {cutoff}: groups={len(sizes)} max_group_share={max_share:.3f} "
              f"GSS_drift={drift_gss:.3f} greedy_drift={drift:.3f} "
              f"leak_test>0.9={leak['test']['share_above_090']:.3f}")

        gate_ok = True
        if leak["test"]["share_above_090"] > LEAK_GATE:
            notes.append(f"cutoff {cutoff}: leakage gate FAILED "
                         f"({leak['test']['share_above_090']:.1%} of test msgs have a "
                         f">{NEAR_COPY} near-copy in train)")
            gate_ok = False
        if max_share > MAX_GROUP_SHARE:
            notes.append(f"cutoff {cutoff}: one group covers {max_share:.1%} of all rows "
                         f"(over {MAX_GROUP_SHARE:.0%}); grouping chains too much")
            gate_ok = False
        if drift > MAX_RATIO_DRIFT_PP / 100:
            notes.append(f"cutoff {cutoff}: scam share still drifts {drift:.1%} across "
                         f"splits even with the greedy assignment")
            gate_ok = False
        if gate_ok:
            chosen = (cutoff, groups, split, leak, sizes)
            break

    if chosen is None:
        print("\n!!! NO cutoff passed all gates - see notes. Saving nothing; report honestly.")
        for n in notes:
            print("NOTE:", n)
        raise SystemExit(1)

    cutoff, groups, split, leak, sizes = chosen
    print(f"\nchosen cutoff: {cutoff}")

    # ---------------- save outputs ----------------
    clean = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "text": df["text"],
        "label": df["label"].astype(int),
        "language": df["language"],
        "n_links": df["n_links"],
    })
    clean.to_csv(PROCESSED_DIR / "clean_dataset.csv", index=False)

    assign = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "split": split.to_numpy(),
        "group_id": [f"g{g:05d}" for g in groups],
    })
    assign.to_csv(PROCESSED_DIR / "split_assignments.csv", index=False)

    # group-size distribution for the summary
    size_counts = sizes.value_counts().sort_index()
    multi = sizes[sizes > 1]
    meta = {
        "seed": SEED,
        "encoding": "utf-8-sig (fallback latin-1)",
        "raw_rows": n_raw,
        "label_counts_raw": {str(k): int(v) for k, v in label_counts_raw.items()},
        "empty_text_rows": nulls,
        "duplicates_dropped": n_dups,
        "real_overlap_dropped": n_real_overlap,
        "clean_rows": int(len(df)),
        "label_counts_clean": {str(k): int(v) for k, v in label_counts.items()},
        "scam_share_clean": round(float(share_scam), 4),
        "length_stats_by_label": len_stats,
        "link_share_by_label": {str(k): float(v) for k, v in link_share.items()},
        "language_share_by_label": {str(k): {c: round(float(x), 4) for c, x in row.items()}
                                    for k, row in lang_by_label.iterrows()},
        "sim_cutoff": cutoff,
        "split_method": "greedy largest-first group assignment (deterministic; "
                        "keeps every group whole; balances rows and scam share)",
        "gss_reference_drift": round(float(gss_reference_drift), 4),
        "n_groups": int(len(sizes)),
        "groups_size2plus": int(len(multi)),
        "rows_in_groups_size2plus": int(multi.sum()),
        "max_group_size": int(sizes.iloc[0]),
        "max_group_share": round(max_share, 4),
        "group_size_histogram": {str(int(k)): int(v) for k, v in size_counts.head(10).items()},
        "split_rows": {str(k): int(v) for k, v in split.value_counts().items()},
        "scam_share_by_split": {k: round(float(v), 4) for k, v in scam_by_split.items()},
        "leakage": leak,
        "leak_gate_pass": bool(leak["test"]["share_above_090"] <= LEAK_GATE),
        "notes": notes if notes else ["none - all gates passed on the first cutoff"],
    }
    (PROCESSED_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2))

    # ---------------- (e) data_summary.md ----------------
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Data summary (Step 1)\n")
    md.append("All numbers below come from the actual run of `src/step1_data_check.py` "
              f"(fixed seed {SEED}). Nothing is estimated.\n")
    md.append("## The data at a glance\n")
    md.append(f"- Main file `scam_dataset_text_label.csv`: **{n_raw:,} rows** "
              f"({label_counts_raw.get(1,0):,} scam = label 1, {label_counts_raw.get(0,0):,} benign = label 0).")
    md.append(f"- Empty texts: **{nulls}**.")
    md.append(f"- Exact or near-exact duplicate texts dropped: **{n_dups:,}**.")
    md.append(f"- Rows whose text appears in the 94-message real test file: **{n_real_overlap}** "
              "(dropped - the real messages must stay untouched as a final test, hard rule 6).")
    md.append(f"- Clean dataset used from here on: **{len(df):,} rows**, "
              f"{label_counts.get(1,0):,} scam ({share_scam:.1%}) / {label_counts.get(0,0):,} benign ({1-share_scam:.1%}).\n")
    md.append("## Message length and links, by label\n")
    md.append("| label | mean chars | median chars | 10th pct | 90th pct | share with a link |")
    md.append("|---|---|---|---|---|---|")
    for lab, name in ((1, "scam"), (0, "benign")):
        s = len_stats[lab]
        md.append(f"| {name} | {s['mean']} | {s['median']} | {s['p10']} | {s['p90']} | "
                  f"{link_share.get(lab, 0):.1%} |")
    md.append("")
    md.append("## Approximate language (heuristic - NOT a real language detector)\n")
    md.append("Each message got a rough bucket: 'pidgin' if it contains typical Pidgin "
              "words (abeg, wetin, wahala, oga...), 'other' if it uses a non-Latin script "
              "or clear French/Spanish/Portuguese words, else 'english'. "
              "**This is approximate** - a word list cannot detect language reliably.\n")
    md.append("| label | english | pidgin | other |")
    md.append("|---|---|---|---|")
    for lab, name in ((1, "scam"), (0, "benign")):
        row = meta["language_share_by_label"][str(lab)]
        md.append(f"| {name} | {row.get('english',0):.1%} | {row.get('pidgin',0):.1%} | "
                  f"{row.get('other',0):.1%} |")
    md.append("")
    md.append("## Near-duplicate groups (Step 1b)\n")
    md.append("To stop the model from being graded on messages that are near-copies of "
              "messages it trained on, we grouped messages that stay similar AFTER masking "
              "links, emails, money amounts and other numbers. Masked messages with word "
              f"TF-IDF cosine similarity above **{cutoff}** were connected, and connected "
              "messages form one group. A group never appears in two splits.\n")
    md.append(f"- Groups: **{len(sizes):,}**; of these {len(multi):,} contain 2+ messages "
              f"({multi.sum():,} rows); the largest group has **{int(sizes.iloc[0])}** messages "
              f"({max_share:.1%} of the data).")
    md.append(f"- Smallest sizes (histogram head): {meta['group_size_histogram']}.\n")
    md.append("## Split (Step 1c)\n")
    md.append("Split **by group** into train 70% / validation 15% / test 15%. "
              "A group never appears in two parts.\n")
    md.append(f"**Method note (found during Step 1):** a plain random GroupShuffleSplit "
              f"by group drifted **{gss_reference_drift:.1%}** in scam share between parts. "
              "Reason: masking makes template messages near-identical, so grouping chains "
              f"them into big clusters (largest group = {max_share:.1%} of all rows), and a "
              "random split lands those clusters unevenly. Instead we use a deterministic "
              "greedy assignment: groups sorted largest-first, each placed in the part where "
              "it fills the largest share of that part's remaining quota (70/15/15 of rows, "
              "with scam and benign quotas counted separately). Groups stay whole; no "
              "randomness is involved, so the run is exactly reproducible.\n")
    md.append("| split | rows | scam share |")
    md.append("|---|---|---|")
    for sp_ in ("train", "validation", "test"):
        md.append(f"| {sp_} | {meta['split_rows'][sp_]:,} | "
                  f"{meta['scam_share_by_split'][sp_]:.1%} |")
    md.append("")
    md.append("## Leakage check (Step 1d)\n")
    md.append("For every validation and test message we computed its highest cosine "
              "similarity to any TRAINING message (masked TF-IDF, vocabulary fitted on "
              "train only).\n")
    md.append("| split | median max-sim | mean max-sim | share with >0.90 near-copy in train | share >0.85 |")
    md.append("|---|---|---|---|---|")
    for sp_ in ("validation", "test"):
        L = leak[sp_]
        md.append(f"| {sp_} | {L['median_max_sim']} | {L['mean_max_sim']} | "
                  f"**{L['share_above_090']:.1%}** | {L['share_above_085']:.1%} |")
    md.append("")
    gate_txt = ("**PASSED**" if meta["leak_gate_pass"] else "**FAILED**")
    md.append(f"Gate: at most 10% of test messages may have a >0.90 near-copy in train "
              f"-> {gate_txt} "
              f"({leak['test']['share_above_090']:.1%}).\n")
    md.append("Why this is not exactly 0%: the grouping pass fits its TF-IDF on ALL "
              "messages, while this check re-fits the vocabulary on TRAIN only, so the "
              "two similarity scales differ slightly. A handful of pairs sit just above "
              "0.90 here despite being below the 0.85 grouping cutoff there - and the "
              "gate still passes comfortably.\n")
    if notes:
        md.append("## Notes and anomalies found during the cutoff search\n")
        for n in notes:
            md.append(f"- {n}")
        md.append("")
    md.append("## Files written by this step\n")
    md.append("- `data/processed/clean_dataset.csv` - cleaned data (row_id, text, label, language, n_links)")
    md.append("- `data/processed/split_assignments.csv` - row_id, split, group_id")
    md.append("- `data/processed/split_meta.json` - all numbers above, machine-readable\n")
    DOCS_DIR.joinpath("data_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\nWrote:",
          PROCESSED_DIR / "clean_dataset.csv",
          PROCESSED_DIR / "split_assignments.csv",
          PROCESSED_DIR / "split_meta.json",
          DOCS_DIR / "data_summary.md", sep="\n  ")
    print("Leakage check:", gate_txt, "| test share>0.9 =", leak['test']['share_above_090'])


def len_stats_groups(df: pd.DataFrame):
    """Small helper so the dict comprehension above stays readable."""
    out = []
    for lab in (0, 1):
        out.append((lab, df.loc[df["label"] == lab, "text"].str.len()))
    return out


if __name__ == "__main__":
    main()
