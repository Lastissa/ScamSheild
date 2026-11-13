# Baseline results (Step 2)

Plain model: word TF-IDF (1-2 grams) + character TF-IDF (3-5 grams) + logistic regression. Trained on `text` only - no link features. All numbers are real outputs of this run, not estimates.

## Test-set comparison

All three runs use the SAME trained model.

## validation (4,447 rows)

| metric | value |
|---|---|
| n | 4447 |
| positives | 3039 |
| tp | 3010 |
| fp | 13 |
| fn | 29 |
| tn | 1395 |
| precision | 0.9957 |
| recall | 0.9905 |
| f1 | 0.9931 |
| roc_auc | 0.9995 |
| pr_auc | 0.9998 |
| false_positive_rate | 0.0092 |
| brier | 0.0219 |
| confusion | tn=1395 fp=13 fn=29 tp=3010 |

## test (4,448 rows)

| metric | value |
|---|---|
| n | 4448 |
| positives | 3039 |
| tp | 3006 |
| fp | 0 |
| fn | 33 |
| tn | 1409 |
| precision | 1.0 |
| recall | 0.9891 |
| f1 | 0.9945 |
| roc_auc | 1.0 |
| pr_auc | 1.0 |
| false_positive_rate | 0.0 |
| brier | 0.0103 |
| confusion | tn=1409 fp=0 fn=33 tp=3006 |

## 94 real scam messages (recall only)

| metric | value |
|---|---|
| n | 94 |
| positives | 94 |
| tp | 88 |
| fp | 0 |
| fn | 6 |
| tn | 0 |
| precision | 1.0 |
| recall | 0.9362 |
| f1 | 0.967 |
| roc_auc | None |
| pr_auc | None |
| false_positive_rate | None |
| brier | 0.0643 |
| confusion | tn=0 fp=0 fn=6 tp=88 |

## By approximate language

Language is a rough word-list heuristic (see data_summary.md); 'other' includes French/Spanish/Portuguese and non-Latin script.

### validation
| language | precision | recall | f1 | pr_auc | false_positive_rate | rows |
|---|---|---|---|---|---|---|
| english | 0.9959 | 0.9868 | 0.9913 | 0.9997 | 0.0085 | 3257 |
| pidgin | 1.0 | 1.0 | 1.0 | None | None | 147 |
| other | 0.9942 | 1.0 | 0.9971 | 1.0 | 0.0113 | 1043 |

### test
| language | precision | recall | f1 | pr_auc | false_positive_rate | rows |
|---|---|---|---|---|---|---|
| english | 1.0 | 0.9832 | 0.9915 | 1.0 | 0.0 | 3201 |
| other | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1247 |

## Reading the numbers

- PR-AUC rewards ranking scams above benign messages; it is the headline number because the data is scam-heavy (68%).
- False-positive rate = share of benign messages wrongly flagged.
- On the 94 real messages we can measure RECALL only (they are all scams), so no false-alarm rate is available for real messages yet.