# Link dependency test (Step 3)

Question: does the model detect scams from the WORDING, or does it just spot scammy links? Three runs compare.

**Run A** - normal: baseline trained and scored on the original text.

**Run B** - links removed at test time: the SAME model as Run A, scored on text where every link was replaced by [URL].

**Run C** - retrained without links: a NEW baseline trained on link-stripped train/validation/test text and scored on stripped text.

All models are the Step-2 baseline: word TF-IDF (1-2 grams) + character TF-IDF (3-5 grams) + logistic regression.

## Results table

| run | set | precision | recall | f1 | PR-AUC | false-positive rate |
|---|---|---|---|---|---|---|
| A | validation | 0.9957 | 0.9905 | 0.9931 | 0.9998 | 0.0092 |
| A | test | 1.0 | 0.9891 | 0.9945 | 1.0 | 0.0 |
| A | real-94 (recall only) | n/a | 0.9362 | n/a | n/a | n/a |
| B | validation | 0.9947 | 0.9852 | 0.9899 | 0.999 | 0.0114 |
| B | test | 1.0 | 0.9852 | 0.9925 | 0.9995 | 0.0 |
| B | real-94 (recall only) | n/a | 0.9362 | n/a | n/a | n/a |
| C | validation | 1.0 | 0.9747 | 0.9872 | 0.9985 | 0.0 |
| C | test | 1.0 | 0.9622 | 0.9807 | 0.9998 | 0.0 |
| C | real-94 (recall only) | n/a | 0.9255 | n/a | n/a | n/a |

## Conclusion

- NO COLLAPSE: Run C stays close to Run A (PR-AUC change 0.00 points if defined; real recall change 0.01). The model learns from wording, not only from links.

- Run A -> real recall 0.936; Run B 0.936; Run C 0.925.
- Run A -> test PR-AUC 1.0; Run B 0.9995; Run C 0.9998.

> A drop is not automatically bad: scam links really do look different > from legitimate ones. A COLLAPSE (>10 points) is the red flag.

## 10 examples: Run A got right, Run C got wrong

These are raw test messages where the normal model (A) predicted correctly but the no-links model (C) erred. They show exactly which messages depend on link signals.

### Example 1 (truth: scam)

- Original: Chevron Nigeria 2026 graduate trainee programme. Apply at https://chevronportal.cc/portal?id=21XuRl. A refundable NGN 3,500 registration fee is required.
- Links in original: ['https://chevronportal.cc/portal?id=21XuRl']
- Stripped: Chevron Nigeria 2026 graduate trainee programme. Apply at [URL]. A refundable NGN 3,500 registration fee is required.
- Run A: probability 0.5673 -> correct
- Run C: probability 0.4366 -> wrong

### Example 2 (truth: scam)

- Original: Hello Zainab, we are from Access Bank. Please send the code you just received so we can reverse the wrong debit of NGN 232,500.
- Stripped: Hello Zainab, we are from Access Bank. Please send the code you just received so we can reverse the wrong debit of NGN 232,500.
- Run A: probability 0.5636 -> correct
- Run C: probability 0.4884 -> wrong

### Example 3 (truth: scam)

- Original: You received NGN 173,500 from Aisha. Claim it before it expires: http://accessbankhelp.vip
- Links in original: ['http://accessbankhelp.vip']
- Stripped: You received NGN 173,500 from Aisha. Claim it before it expires: [URL]
- Run A: probability 0.5091 -> correct
- Run C: probability 0.265 -> wrong

### Example 4 (truth: scam)

- Original: Chevron Nigeria 2026 graduate trainee programme. Apply at http://chevr0n.com. A refundable NGN 8,500 registration fee is required.
- Links in original: ['http://chevr0n.com']
- Stripped: Chevron Nigeria 2026 graduate trainee programme. Apply at [URL]. A refundable NGN 8,500 registration fee is required.
- Run A: probability 0.6149 -> correct
- Run C: probability 0.4313 -> wrong

### Example 5 (truth: scam)

- Original: Hello Zainab, we are from Access Bank. Please send the code you just received so we can reverse the wrong debit of NGN 39,500.
- Stripped: Hello Zainab, we are from Access Bank. Please send the code you just received so we can reverse the wrong debit of NGN 39,500.
- Run A: probability 0.5682 -> correct
- Run C: probability 0.4949 -> wrong

### Example 6 (truth: scam)

- Original: Dangote Nigeria 2026 graduate trainee programme. Apply at http://dangote-confirm.com/confirm. A refundable NGN 13,000 registration fee is required.
- Links in original: ['http://dangote-confirm.com/confirm']
- Stripped: Dangote Nigeria 2026 graduate trainee programme. Apply at [URL]. A refundable NGN 13,000 registration fee is required.
- Run A: probability 0.8082 -> correct
- Run C: probability 0.4703 -> wrong

### Example 7 (truth: scam)

- Original: Chevron Nigeria 2026 graduate trainee programme. Apply at tinyurl.com/CZEWtvN. A refundable NGN 11,500 registration fee is required.
- Links in original: ['tinyurl.com/CZEWtvN']
- Stripped: Chevron Nigeria 2026 graduate trainee programme. Apply at [URL]. A refundable NGN 11,500 registration fee is required.
- Run A: probability 0.6476 -> correct
- Run C: probability 0.4581 -> wrong

### Example 8 (truth: scam)

- Original: Hello Damilola, we are from Wema Bank. Please send the code you just received so we can reverse the wrong debit of NGN 58,000.
- Stripped: Hello Damilola, we are from Wema Bank. Please send the code you just received so we can reverse the wrong debit of NGN 58,000.
- Run A: probability 0.5173 -> correct
- Run C: probability 0.4425 -> wrong

### Example 9 (truth: scam)

- Original: You received NGN 43,000 from Hauwa. Claim it before it expires: http://secure-uba.com
- Links in original: ['http://secure-uba.com']
- Stripped: You received NGN 43,000 from Hauwa. Claim it before it expires: [URL]
- Run A: probability 0.5049 -> correct
- Run C: probability 0.2883 -> wrong

### Example 10 (truth: scam)

- Original: Dangote Nigeria 2026 graduate trainee programme. Apply at http://dangote-ng.com. A refundable NGN 3,500 registration fee is required.
- Links in original: ['http://dangote-ng.com']
- Stripped: Dangote Nigeria 2026 graduate trainee programme. Apply at [URL]. A refundable NGN 3,500 registration fee is required.
- Run A: probability 0.6407 -> correct
- Run C: probability 0.4452 -> wrong
