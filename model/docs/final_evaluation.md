# Final evaluation (Step 5)

All numbers are real outputs of `src/step5_final_evaluation.py` on artifacts/scam_model.joblib. Nothing is estimated.

## Headline results

## validation

| metric | value |
|---|---|
| rows | 4447 |
| scam rows | 3039 |
| precision | 0.9962 |
| recall | 0.9599 |
| F1 | 0.9777 |
| ROC-AUC | 0.9944 |
| PR-AUC | 0.9978 |
| false-positive rate | 0.0078 |
| Brier score | 0.0325 |
| confusion (tn, fp, fn, tp) | [1397, 11, 122, 2917] |

## test

| metric | value |
|---|---|
| rows | 4448 |
| scam rows | 3039 |
| precision | 0.9993 |
| recall | 0.9579 |
| F1 | 0.9782 |
| ROC-AUC | 0.9991 |
| PR-AUC | 0.9997 |
| false-positive rate | 0.0014 |
| Brier score | 0.0237 |
| confusion (tn, fp, fn, tp) | [1407, 2, 128, 2911] |

## 94 real scam messages (recall only, threshold 0.5)

| metric | value |
|---|---|
| rows | 94 |
| scam rows | 94 |
| precision | 1.0 |
| recall | 0.6702 |
| F1 | 0.8025 |
| ROC-AUC | None |
| PR-AUC | None |
| false-positive rate | None |
| Brier score | 0.1871 |
| confusion (tn, fp, fn, tp) | [0, 0, 31, 63] |

## 94 real scam messages at operating threshold (0.334)

| metric | value |
|---|---|
| rows | 94 |
| scam rows | 94 |
| precision | 1.0 |
| recall | 0.9468 |
| F1 | 0.9727 |
| ROC-AUC | None |
| PR-AUC | None |
| false-positive rate | None |
| Brier score | 0.1871 |
| confusion (tn, fp, fn, tp) | [0, 0, 5, 89] |

## real_legit_messages.csv
File not present -> false-alarm rate on real legitimate messages **was NOT measured**, per the project spec.

## Scores by approximate language

Language is a rough word-list heuristic (see data_summary.md).

### validation
| language | rows | precision | recall | f1 | PR-AUC | false-positive rate |
|---|---|---|---|---|---|---|
| english | 3257 | 0.9957 | 0.9446 | 0.9695 | 0.9971 | 0.0085 |
| pidgin | 147 | 1.0 | 1.0 | 1.0 | None | None |
| other | 1043 | 0.9971 | 1.0 | 0.9985 | 1.0 | 0.0056 |

### test
| language | rows | precision | recall | f1 | PR-AUC | false-positive rate |
|---|---|---|---|---|---|---|
| english | 3201 | 0.9989 | 0.9358 | 0.9663 | 0.9993 | 0.0016 |
| other | 1247 | 1.0 | 0.9981 | 0.9991 | 1.0 | 0.0 |

## Risk thresholds (chosen on validation, Step 6)

- high risk: score >= 0.334 (validation precision at/above: 0.9516, target >= 0.95)
- low risk: score <= 0.333 (>= 0.99 of validation scams score above it; actual 0.9918)
- medium: between 0.333 and 0.334.
The score is a *risk score*, not a real-world probability: the training data has far more scams than real life.

## 20 most influential features (interpretation model, uncalibrated LR)

Features with the largest positive coefficients push toward 'scam'; largest negative coefficients push toward 'benign'. Computed with an uncalibrated logistic regression on the same features, for reading only.

### For scam
| feature | coefficient |
|---|---|
| links__links__hyphen_host | 3.1174 |
| text__word__to | 2.7254 |
| links__links__is_shortener | 2.4475 |
| links__links__tld_risk | 1.6902 |
| text__word__now | 1.6245 |
| text__word__fee | 1.5113 |
| text__word__de | 1.446 |
| text__word__need | 1.4417 |
| text__word__confirm | 1.4367 |
| text__word__http | 1.3916 |
| text__word__id | 1.3597 |
| text__word__pay | 1.3244 |
| text__word__and | 1.2627 |
| text__word__pay ngn | 1.1811 |
| text__word__please | 1.1669 |
| text__word__send | 1.144 |
| text__word__verify | 1.1064 |
| text__word__deposit | 1.1059 |
| text__char__ i  | 1.0933 |
| text__word__we | 1.046 |

### For benign
| feature | coefficient |
|---|---|
| text__word__https | -2.011 |
| text__word__app | -1.5821 |
| text__word__any | -1.4537 |
| text__word__amt ngn | -1.3215 |
| text__word__amt | -1.3215 |
| text__word__token | -1.3067 |
| text__word__kwh | -1.3067 |
| text__word__any time | -1.2928 |
| text__word__no | -1.2841 |
| text__word__careers | -1.2753 |
| text__word__apply | -1.2695 |
| text__word__is | -1.2346 |
| text__word__on | -1.2318 |
| text__word__meter | -1.2163 |
| text__word__view | -1.211 |
| links__links__n_subdomains | -1.042 |
| text__word__at https | -0.9859 |
| text__word__is ngn | -0.9741 |
| text__word__the app | -0.9722 |
| text__word__bluepeak | -0.9646 |

## 15 raw errors the model made (test split, threshold 0.5)

### Example 1 (scam, probability 0.4753)
- Meridian Financial: detectamos atividade incomum em sua conta. Verifique sua identidade agora em https://meridianfinnancia.net/support ou ligue para +1-305-555-0121, ou sua conta será suspensa hoje.

### Example 2 (scam, probability 0.4463)
- Ledger & Co. Bank: detectamos atividade incomum em sua conta. Verifique sua identidade agora em https://ledgercboank.com/bvn ou ligue para +1-202-555-0172, ou sua conta será suspensa em 24 horas.

### Example 3 (benign, probability 0.5041)
- Falcon Parcel Network: delivery completed, left at your front door. Track your package any time using the reference number above on our website. Contact support through the app if anything looks wrong. This is an automated notification. Track it here: https://www.falcon-parcel-network.ng/track

### Example 4 (benign, probability 0.5948)
- Delivery completed, left at your front door. No signature was required for this delivery. Contact support through the app if anything looks wrong. Thanks for choosing Falcon Parcel Network. Track it here: https://rb.gy/5x6aUGY - Falcon Parcel Network

### Example 5 (scam, probability 0.3486)
- Chevron Nigeria 2026 graduate trainee programme. Apply at https://chevronportal.cc/portal?id=21XuRl. A refundable NGN 3,500 registration fee is required.

### Example 6 (scam, probability 0.3634)
- Hello Zainab, we are from Access Bank. Please send the code you just received so we can reverse the wrong debit of NGN 232,500.

### Example 7 (scam, probability 0.3338)
- You received NGN 173,500 from Aisha. Claim it before it expires: http://accessbankhelp.vip

### Example 8 (scam, probability 0.3551)
- Chevron Nigeria 2026 graduate trainee programme. Apply at http://chevr0n.com. A refundable NGN 8,500 registration fee is required.

### Example 9 (scam, probability 0.3766)
- Hello Ibrahim, we are from First Bank. Please send the code you just received so we can reverse the wrong debit of NGN 136,500.

### Example 10 (scam, probability 0.3527)
- Hello Temitope, we are from Wema Bank. Please send the code you just received so we can reverse the wrong debit of NGN 232,500.

### Example 11 (scam, probability 0.3673)
- Hello Zainab, we are from Access Bank. Please send the code you just received so we can reverse the wrong debit of NGN 39,500.

### Example 12 (scam, probability 0.484)
- UBA Alert: Your ATM card has been blocked. Reactivate immediately: https://ubagroup.com.secure-yxw5.com/claim

### Example 13 (scam, probability 0.3671)
- Hello Damilola, we are from Wema Bank. Please send the code you just received so we can reverse the wrong debit of NGN 58,000.

### Example 14 (scam, probability 0.3383)
- You received NGN 43,000 from Hauwa. Claim it before it expires: http://secure-uba.com

### Example 15 (scam, probability 0.3892)
- Chevron Nigeria 2026 graduate trainee programme. Apply at https://rb.gy/dA27OJs. A refundable NGN 18,000 registration fee is required.

## Template memorization check

- Test PR-AUC 0.9997 is above 0.99 while real-message recall at the operating threshold is 0.9468 (gap 0.053). Synthetic scores are near-perfect; real-message performance is lower, consistent with the data being mostly synthetic templates. At a fixed 0.5 threshold recall is 0.6702 because calibrated probabilities sit in separated clusters - the operating threshold 0.334 is the meaningful one.
