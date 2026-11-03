# Model Card

## Name

PMI + length-control hallucination stacker, version `2026-09-tokenfix-length-control`.

## Intended Use

Rank and classify short question-answer pairs as factual or hallucinated from precomputed detector features. The intended use is review prioritisation, audit triage, or research comparison.

## Inputs

The model consumes four numeric features:

- `likelihood_risk`
- `likelihood_token_count_log`
- `contradiction_risk`
- `finch_risk`

It does not consume raw text directly. Raw Q/A feature generation must be performed upstream by the detector pipeline.

## Output

The model emits:

- `ensemble_risk`: calibrated logistic stacker score.
- `ensemble_prediction`: `1` when `ensemble_risk >= 0.4371241698739282`, otherwise `0`.

## Training and Calibration

The stacker is a `StandardScaler` plus L2 logistic regression with `C = 0.01`. The threshold was selected on 600 validation out-of-fold rows using `max F1 subject to precision >= 0.60`.

## Held-Out Performance

Balanced 1,500-row held-out test, 500 rows each from FactCHD, HaluEvalQA, and TruthfulQA:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.670 |
| Precision | 0.631 |
| Recall | 0.817 |
| F1 | 0.712 |
| AUROC | 0.740 |
| AUPRC | 0.697 |

Confusion matrix, labels `[factual, hallucination]`: `[[392, 358], [137, 613]]`.

## Limitations

The model is not evidence-grounded and cannot verify facts against external sources. The answer-length control improves calibration but also shows that benchmark style contributes signal; performance should be revalidated on any new domain.
