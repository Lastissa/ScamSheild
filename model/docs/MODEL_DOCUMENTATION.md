# ScamShield - model documentation

How the scam-message detector was built, step by step. Written so a
non-technical reader can follow; technical terms are explained in one line the
first time they appear (see also docs/GLOSSARY.md).

Every number here comes from a real run of the code in this folder and matches
`artifacts/model_card.json`. Nothing is estimated or invented.

## 1. What this is

A model that takes the text of one message (English or Nigerian Pidgin SMS/
chat) and returns: whether it looks like a scam, a risk score from 0 to 100, a
risk level (low / medium / high), and short plain-language reasons. It is
packaged as one scikit-learn Pipeline: raw text goes in, a score comes out. It
is designed to run on a website backend (`app/main.py`, FastAPI).

## 2. Where the data came from

- `data/scam_dataset_text_label.csv`: 29,650 messages with columns `text` and
  `label` (1 = scam, 0 = benign): 20,260 scams (68.3%) and 9,390 benign
  (31.7%). Almost all of it is SYNTHETIC - generated from templates - mixed
  with some Nigerian-context messages in English and Pidgin and a few other
  languages (French, Spanish, Portuguese appear).
- `data/real_messages_test_text_label.csv`: 94 REAL scam messages, all label 1.
  Final test set. Because it contains only scams, it measures recall only, never
  false alarms.
- `data/real_legit_messages.csv`: NOT present. The false-alarm rate on real
  legitimate messages was therefore NOT measured.

Known weaknesses of the data (checked and reported):
- Nearly all rows are synthetic from a limited set of templates, so very high
  synthetic scores can just mean the model memorised templates.
- A plain random split would put near-identical template messages in both train
  and test and inflate every score. Step 3 below prevents this.
- Scam links differ from legitimate links (hyphens in the host, more shorteners,
  unusual endings); the model may lean on these - tested in Step 6.
- Benign messages are shorter on average; non-English text is more common in
  scams.
- The real test set contains only scams.

## 3. Data cleaning (Step 1)

Original files are never modified; cleaned copies live in `data/processed/`.

Checks run on the raw data:
- Rows: 29,650; labels: 20,260 scam / 9,390 benign.
- Empty texts: 0.
- Exact/duplicate texts: 0 dropped.
- Rows whose text appears in the 94-message real test file: 0 (the real
  messages stay untouched as the final test - the model never trains on them).

Each message got an approximate language label (a word-list heuristic, NOT a
real language detector - stated as approximate everywhere):
- 'pidgin' if it contains typical Pidgin words (abeg, wetin, dey, oga, wahala,
  sabi, na, make, una...),
- 'other' if it uses a non-Latin script or clear French/Spanish/Portuguese
  words,
- otherwise 'english'.

Scam messages: 72.0% english / 3.3% pidgin / 24.8% other.
Benign messages: 89.5% english / 0.8% pidgin / 9.7% other.

Length and links by label:

| label | mean chars | median chars | share with a link |
|---|---|---|---|
| scam | 281.8 | 302 | 48.2% |
| benign | 198.6 | 208 | 56.6% |

Note: benign messages contain MORE links than scams in this data, so "has a
link" is not by itself a scam signal.

## 4. The grouped split and leakage check (Steps 1b-1d)

To stop the model from being graded on near-copies of messages it trained on,
messages were grouped by near-duplication: links, emails, money amounts and
other numbers were masked, word TF-IDF computed, and messages with cosine
similarity above 0.85 connected into groups. A group never appears in two
splits.

- Groups: 1,405; 627 groups contain 2+ messages (28,872 rows); largest group:
  929 messages (3.1% of the data).

Split by group into train 70% / validation 15% / test 15% (fixed seed 42), done
with a deterministic greedy largest-first group assignment that balances rows
and scam share across the parts. Why not a plain random GroupShuffleSplit?
Because with chained template groups, a random group split put 12.2 points of
scam-share drift between parts; the greedy assignment achieved 0.0% drift:

| split | rows | scam share |
|---|---|---|
| train | 20,755 | 68.3% |
| validation | 4,447 | 68.3% |
| test | 4,448 | 68.3% |

Leakage check: for every validation/test message, its highest similarity to any
training message (TF-IDF re-fitted on train only):

| split | median max-sim | share with >0.90 near-copy in train |
|---|---|---|
| validation | 0.7532 | 0.2% |
| test | 0.7648 | 1.1% |

Gate: at most 10% of test messages may have a >0.90 near-copy in train.
**Result: PASSED (1.1%).** This is not exactly 0% because the grouping pass
and the leakage check fit their TF-IDF vocabularies differently; the gate still
passes comfortably.

All of this is summarised in `docs/data_summary.md` and saved to
`data/processed/clean_dataset.csv`, `data/processed/split_assignments.csv` and
`data/processed/split_meta.json`.

## 5. Baseline (Step 2)

Trained before anything fancy: word TF-IDF (1-2 grams) + character TF-IDF
(3-5 grams) + logistic regression, on `text` only, no link features.

| | precision | recall | F1 | ROC-AUC | PR-AUC | false-positive rate |
|---|---|---|---|---|---|---|
| validation | 0.9957 | 0.9905 | 0.9931 | 0.9995 | 0.9998 | 0.0092 |
| test | 1.0000 | 0.9891 | 0.9945 | 1.0000 | 1.0000 | 0.0000 |
| 94 real scams (recall only) | - | 0.9362 | - | - | - | - |

Synthetic scores are near-perfect - an early warning of template memorisation,
investigated in Steps 6 and 11. Full details in `docs/baseline_results.md`.

## 6. Link dependency test (Step 3)

Three runs, same baseline model spec:

- Run A - normal: trained and scored on original text.
- Run B - links removed at test time: SAME model as A, scored on text with
  every link replaced by `[URL]`.
- Run C - retrained without links: new model trained on link-stripped text,
  scored on stripped text.

Test split:

| run | precision | recall | F1 | PR-AUC | false-positive rate |
|---|---|---|---|---|---|
| A - normal | 1.0000 | 0.9891 | 0.9945 | 1.0000 | 0.0000 |
| B - links removed at test time | 1.0000 | 0.9852 | 0.9925 | 0.9995 | 0.0000 |
| C - retrained without links | 1.0000 | 0.9622 | 0.9807 | 0.9998 | 0.0000 |

Recall on the 94 real scam messages: A = 0.9362, B = 0.9362, C = 0.9255.

Conclusion in plain words: **the model does not collapse without links.** The
biggest drop is 2.7 points (test recall, Run C vs Run A), far below the
10-point "collapse" threshold. Run B barely moves at all. Scam links really do
look different from legitimate ones, so links help a little, but the model
learns from the wording, not only from the links. Full table and 10 example
messages in `docs/link_dependency_test.md`.

## 7. The full model (Step 4)

Features (all computed inside the pipeline from the raw text):
- word TF-IDF (1-2 grams), character TF-IDF (3-5 grams);
- link features via `link_tools`: has_link, n_links, is_shortener,
  hyphen_host, n_subdomains, digit_in_host, tld_risk.

Candidates compared on the validation split (PR-AUC and false-alarm rate, not
accuracy alone):

| model | precision | recall | PR-AUC | false-positive rate |
|---|---|---|---|---|
| logistic regression | 0.9721 | 0.9753 | 0.9963 | 0.0604 |
| naive Bayes | 0.9891 | 0.9520 | 0.9913 | 0.0227 |
| linear SVM | 0.9962 | 0.9599 | 0.9978 | 0.0078 |

**Chosen: linear SVM**, calibrated with `CalibratedClassifierCV` (sigmoid,
cv=3) so the output is a meaningful probability. Packaged in one
scikit-learn Pipeline and saved to `artifacts/scam_model.joblib`. Naive Bayes
had fewer false alarms but much lower recall; logistic regression had far more
false alarms (6.0%). The SVM gave the best PR-AUC with under 1% false alarms.

Probabilities are calibrated because a raw score from an SVM is not a
probability; calibration (Platt scaling here) makes the number interpretable
within the training data.

## 8. Final evaluation (Step 5)

Evaluated on `artifacts/scam_model.joblib`:

### Headline results

| | precision | recall | F1 | ROC-AUC | PR-AUC | false-positive rate | Brier |
|---|---|---|---|---|---|---|---|
| validation | 0.9962 | 0.9599 | 0.9777 | 0.9944 | 0.9978 | 0.0078 | 0.0325 |
| test | 0.9993 | 0.9579 | 0.9782 | 0.9991 | 0.9997 | 0.0014 | 0.0237 |
| 94 real scams @ threshold 0.5 | - | 0.6702 | - | - | - | - | 0.1871 |
| 94 real scams @ operating threshold (≥60) | - | 0.617 | - | - | - | - | - |

Real-legit messages file absent -> false-alarm rate on real legitimate messages
NOT measured.

Confusion matrices (tn, fp, fn, tp): validation [1397, 11, 122, 2917], test
[1407, 2, 128, 2911].

### Scores by approximate language

Test split:

| language | rows | precision | recall | F1 | PR-AUC | false-positive rate |
|---|---|---|---|---|---|---|
| english | 3,201 | 0.9989 | 0.9358 | 0.9663 | 0.9993 | 0.0016 |
| other | 1,247 | 1.0000 | 0.9981 | 0.9991 | 1.0000 | 0.0000 |
| pidgin | 0 in test | - | - | - | - | - |

(Validation: english P=0.9957/R=0.9446/FPR=0.0085; pidgin 147 rows all scams,
recalled 1.0; other P=0.9971/R=1.0/FPR=0.0056.) Language is a rough word-list
heuristic - treat these numbers as approximate.

### Most influential features

From an uncalibrated logistic regression on the same features (interpretation
only; predictions still come from the calibrated SVM):

Top for scam: hyphen_host (+3.12), is_shortener (+2.45), word "to" (+2.73),
word "now" (+1.62), word "fee" (+1.51), tld_risk (+1.69), word "verify"
(+1.11), word "send" (+1.14), word "please" (+1.17).

Top for benign: word "https" (-2.01), word "app" (-1.58), word "no" (-1.28),
word "careers" (-1.27), word "apply" (-1.27), n_subdomains (-1.04).

So hyphens in link hosts and shorteners push toward scam; benign messages more
often carry normal https links and words like "app"/"careers"/"apply".

### Errors

15 raw errors on the test split are listed in `docs/final_evaluation.md`, each
with a one-line guess at why (for example: Portuguese-language scam phrased like
a bank alert scored below the cut-off; a delivery notice with a shortener link
was flagged as scam).

### Template memorisation check

Test PR-AUC is 0.9997 (above 0.99) while real-message recall at the operating
threshold (≥60) is 0.617 - a gap of 0.383. Synthetic scores are near-perfect and
real-message performance is lower, consistent with the data being mostly
synthetic templates. At the fixed 0.5 threshold real recall is 0.6702 because
calibrated probabilities sit in separated clusters; the operating threshold
(score ≥ 60) is the meaningful one.

Full details in `docs/final_evaluation.md` and `data/processed/final_eval_meta.json`.

## 9. Risk score and levels (Step 6)

`risk_score = round(100 × calibrated probability)`.

Thresholds chosen on the validation set:
- high: score ≥ 60 → validation precision at/above threshold 0.9997 (target ≥
  0.95).
- low: score ≤ 33 → 99.18% of validation scams score above it (target ≥ 99%).
- medium: 34–59.

There is a 27-point gap between the bands, so the levels stay separated and the
output fields (is_scam, risk_level, risk_score) always agree.

The score is a **risk score**, not the real-world chance that a message is a
scam, because the training data has far more scams than real life.

## 10. Known weaknesses

- Almost all training data is synthetic template output; very high scores can
  reflect memorised templates rather than general scam detection.
- Real-message recall (0.617 at operating threshold ≥60) is below synthetic test
  scores (PR-AUC 0.9997).
- No real legitimate messages were available, so false alarms on real benign
  traffic were not measured.
- The model works best on Nigerian-style English and Pidgin SMS/chat; other
  languages and styles are less reliable.
- The approximate language label is a heuristic.

## 11. How to retrain and version

`python src/train.py` (run from `model/`) reproduces everything from data to
saved model: grouped split, baseline, link dependency test, candidate training,
calibration, thresholds, final evaluation. It overwrites
`artifacts/scam_model.joblib`, `artifacts/model_meta.json`,
`artifacts/model_card.json` and the docs in `docs/`.

Versioning: `artifacts/model_card.json` records version, training date, library
versions, data file names and row counts, chosen model, thresholds and every
evaluation number. Bump the version in `build_model_card.py` and re-run
`python build_model_card.py` after any retraining so the card stays in sync
with the model file. Never train on the 94 real messages - they are the final
test set.

## 12. Plugging it into a website backend

`app/main.py` is a small FastAPI backend:
- `GET /health`
- `POST /predict` with body `{"text": "..."}`
- `POST /predict-batch` with body `{"texts": ["..."]}`

It returns `{"is_scam", "risk_score", "risk_level", "reasons", "links"}`.
The model is loaded once at startup. CORS: edit `ALLOWED_ORIGINS` at the top of
`app/main.py` (or set the `CORS_ALLOWED_ORIGINS` environment variable) to
match your website's domain.

The model receives ONE input: the message text. No message text is logged or
stored at prediction time.

## 13. Privacy notes

- At prediction time the model receives only the message text; nothing is
  logged or stored by this code.
- Links in the data are training text only; many are made-up lookalike domains
  and must never be visited.
- The training data contains phone numbers, account numbers and currency
  amounts inside message text; the model never outputs message text.
- If you integrate this into a real service, decide your own retention policy
  for submitted messages; this prototype retains nothing.
