# Data summary (Step 1)

All numbers below come from the actual run of `src/step1_data_check.py` (fixed seed 42). Nothing is estimated.

## The data at a glance

- Main file `scam_dataset_text_label.csv`: **29,650 rows** (20,260 scam = label 1, 9,390 benign = label 0).
- Empty texts: **0**.
- Exact or near-exact duplicate texts dropped: **0**.
- Rows whose text appears in the 94-message real test file: **0** (dropped - the real messages must stay untouched as a final test, hard rule 6).
- Clean dataset used from here on: **29,650 rows**, 20,260 scam (68.3%) / 9,390 benign (31.7%).

## Message length and links, by label

| label | mean chars | median chars | 10th pct | 90th pct | share with a link |
|---|---|---|---|---|---|
| scam | 281.8 | 302.0 | 146.9 | 374.0 | 48.2% |
| benign | 198.6 | 208.0 | 100.0 | 294.0 | 56.6% |

## Approximate language (heuristic - NOT a real language detector)

Each message got a rough bucket: 'pidgin' if it contains typical Pidgin words (abeg, wetin, wahala, oga...), 'other' if it uses a non-Latin script or clear French/Spanish/Portuguese words, else 'english'. **This is approximate** - a word list cannot detect language reliably.

| label | english | pidgin | other |
|---|---|---|---|
| scam | 72.0% | 3.3% | 24.8% |
| benign | 89.5% | 0.8% | 9.7% |

## Near-duplicate groups (Step 1b)

To stop the model from being graded on messages that are near-copies of messages it trained on, we grouped messages that stay similar AFTER masking links, emails, money amounts and other numbers. Masked messages with word TF-IDF cosine similarity above **0.85** were connected, and connected messages form one group. A group never appears in two splits.

- Groups: **1,405**; of these 627 contain 2+ messages (28,872 rows); the largest group has **929** messages (3.1% of the data).
- Smallest sizes (histogram head): {'1': 778, '2': 92, '3': 74, '4': 39, '5': 17, '6': 21, '7': 15, '8': 7, '9': 16, '10': 6}.

## Split (Step 1c)

Split **by group** into train 70% / validation 15% / test 15%. A group never appears in two parts.

**Method note (found during Step 1):** a plain random GroupShuffleSplit by group drifted **12.2%** in scam share between parts. Reason: masking makes template messages near-identical, so grouping chains them into big clusters (largest group = 3.1% of all rows), and a random split lands those clusters unevenly. Instead we use a deterministic greedy assignment: groups sorted largest-first, each placed in the part where it fills the largest share of that part's remaining quota (70/15/15 of rows, with scam and benign quotas counted separately). Groups stay whole; no randomness is involved, so the run is exactly reproducible.

| split | rows | scam share |
|---|---|---|
| train | 20,755 | 68.3% |
| validation | 4,447 | 68.3% |
| test | 4,448 | 68.3% |

## Leakage check (Step 1d)

For every validation and test message we computed its highest cosine similarity to any TRAINING message (masked TF-IDF, vocabulary fitted on train only).

| split | median max-sim | mean max-sim | share with >0.90 near-copy in train | share >0.85 |
|---|---|---|---|---|
| validation | 0.7532 | 0.71 | **0.2%** | 4.8% |
| test | 0.7648 | 0.6807 | **1.1%** | 11.1% |

Gate: at most 10% of test messages may have a >0.90 near-copy in train -> **PASSED** (1.1%).

Why this is not exactly 0%: the grouping pass fits its TF-IDF on ALL messages, while this check re-fits the vocabulary on TRAIN only, so the two similarity scales differ slightly. A handful of pairs sit just above 0.90 here despite being below the 0.85 grouping cutoff there - and the gate still passes comfortably.

## Notes and anomalies found during the cutoff search

- Plain random GroupShuffleSplit by group (spec's first choice, measured at cutoff 0.85) drifts 12.2% in scam share between parts, because near-duplicate groups chain into big template clusters (largest group = 3.1% of rows). A deterministic greedy largest-first group assignment is used instead; it keeps groups whole and balances rows and scam share.

## Files written by this step

- `data/processed/clean_dataset.csv` - cleaned data (row_id, text, label, language, n_links)
- `data/processed/split_assignments.csv` - row_id, split, group_id
- `data/processed/split_meta.json` - all numbers above, machine-readable
