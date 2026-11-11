# NimbusPay Support Ticket Triage: Fine-Tuning Report

## 1. Objective and approach

The objective was to fine-tune a small instruction model to classify NimbusPay support tickets and return exactly eight JSON fields. The base model was `Qwen/Qwen2.5-1.5B-Instruct` (approximately 1.54B parameters). Fine-tuning used PEFT QLoRA with 4-bit NF4 quantization on a Google Colab NVIDIA Tesla T4 GPU.

The final selected adapter is Experiment 3, using LoRA rank 16 and alpha 32. It has 18,464,768 trainable parameters (1.182% of the base model). The final adapter is provided in `adapter/`.

## 2. Data audit and preparation

The original training set contained 1,445 records. The audit identified malformed/truncated labels, legacy priority keys, priority inconsistencies, date-format issues, transaction-ID formatting issues, and duplicate messages.

| Audit finding | Records |
|---|---:|
| Malformed/truncated labels | 25 |
| Incorrect/legacy priority keys | 70 |
| Incorrect transaction dates | 30 |
| Incorrect transaction IDs | 20 |
| Duplicate records identified | 60 |
| Final cleaned training records | 1,360 |

Malformed labels were excluded rather than reconstructed. Duplicate records were removed. Legacy priority fields were migrated and priorities recalculated against the supplied schema. Dates and transaction IDs were normalized. Additional manual corrections restored 57 escalation labels that had been incorrectly changed by an earlier cleaning rule and repaired two P1 cases. The changes are recorded in `cleaning_log.csv`.

The final cleaned dataset contains 1,360 records. The 200-record development set was kept separate from training. The 400 test records were used only for final inference.

Sequence-length analysis gave a median of 168 tokens, 95th percentile of approximately 225 tokens, and maximum of 2,215 tokens. A maximum sequence length of 2,304 was selected, so no training example was truncated. The final tokenized dataset contained 87,285 supervised assistant tokens and zero empty assistant targets. Prompt tokens were masked with label `-100`; loss was computed only on assistant targets.

## 3. Baselines and experiments

The base-model zero-shot baseline produced 0% valid JSON, 0% exact match, and 0% mean field accuracy on the 200 development records. A separate base-model baseline with schema rules embedded in the prompt was not completed.

| Run | Training data | Configuration | Dev exact match | Mean field accuracy | Peak GPU memory | Training time |
|---|---|---|---:|---:|---:|---:|
| Experiment 1 | Earlier cleaned version | QLoRA, rank 16, 2 epochs | 79.5% | 96.50% | 9.17 GB | 32.18 min |
| Experiment 2 | Corrected cleaned data | QLoRA, rank 8, 2 epochs | 56.5% | 92.875% | 9.44 GB | 32.63 min |
| Experiment 3 | Corrected cleaned data | QLoRA, rank 16, 2 epochs | **81.0%** | **96.625%** | 9.415 GB | 33.35 min |

All fine-tuning runs used a learning rate of 2e-4, effective batch size 8, 340 optimizer steps, warmup of 17 steps, cosine scheduling, FP16 compute, gradient checkpointing, and 8-bit AdamW. Experiment 2 and Experiment 3 form the controlled rank comparison: rank 8 versus rank 16, with the corrected dataset and other settings held constant.

Experiment 1 used an earlier cleaned dataset containing incorrect escalation labels. Its score is retained for transparency but should not be treated as a fair comparison against Experiments 2 and 3.

The assignment requested at least four controlled experiments, including raw versus cleaned data and 16-bit LoRA versus 4-bit QLoRA. Those comparisons, a fourth experiment, and the prompt-rules baseline were not completed within the available compute time. No results are fabricated for these missing experiments.

## 4. Best model results and error analysis

Experiment 3 achieved 100% JSON validity, 81.0% exact match (162/200), and 96.625% mean field accuracy.

| Field | Accuracy |
|---|---:|
| Category | 98.0% |
| Priority | 86.0% |
| Amount (paise) | 96.5% |
| Transaction date | 98.5% |
| Transaction ID | 100.0% |
| Channel | 100.0% |
| Language | 100.0% |
| Needs human | 94.0% |

The strongest categories were KYC (100% exact match) and other (93.75%). Refund achieved 78.18%, payment_failed 71.70%, fraud 80.0%, offers 77.78%, and account_access 84.21%.

There were 38 incorrect predictions in the development set. The first ten observed failures were:

| Ticket | Incorrect field(s) | Observed error |
|---|---|---|
| dv-00005 | priority | P2 predicted as P3 |
| dv-00011 | priority, needs_human | P1 predicted as P2; true predicted false |
| dv-00016 | priority, needs_human | P2 predicted as P1; false predicted true |
| dv-00021 | priority | P2 predicted as P3 |
| dv-00025 | amount_paise | 2,400 predicted as 24 |
| dv-00027 | amount_paise | 400,000 predicted as 40,000 |
| dv-00034 | priority, needs_human | P2 predicted as P1; false predicted true |
| dv-00037 | priority, needs_human | P1 predicted as P3; true predicted false |
| dv-00040 | priority, needs_human | P2 predicted as P1; false predicted true |
| dv-00053 | priority | P2 predicted as P3 |

Inspection of the available ticket texts showed inconsistent handling of shorthand amount expressions such as `24` and `4k`, and inconsistent application of priority thresholds. Priority mistakes also propagated into `needs_human`, because P1 cases require human escalation. These examples indicate that numeric normalization and rule-boundary reasoning are the main areas for improvement. Transaction ID, channel, and language extraction were perfect on the development set.

## 5. Final test inference

The selected Experiment 3 adapter generated 400 predictions for the unlabelled test set. The saved file is `predictions/predictions_test.jsonl`.

| Validation check | Result |
|---|---:|
| Total predictions | 400 |
| Duplicate IDs | 0 |
| Invalid JSON | 0 |
| Invalid schema | 0 |
| Invalid category/priority | 0 |
| Invalid dates | 1 |

Ticket `ts-00001` contains the model-generated date `2026-07-00`, which is not a valid calendar date. The prediction was not manually altered because the inference instructions permit only whitespace stripping after generation. The original raw model output is retained in the submission.

## 6. Limitations and next steps

The experiment budget did not allow all requested controlled comparisons. The results are based on one development split, and the test set has no labels available for local scoring. Future work should prioritize a raw-versus-cleaned data comparison, 16-bit LoRA versus QLoRA, a learning-rate or epoch sweep, and a schema-rules baseline. Further error-focused data review should target priority thresholds, escalation conditions, and rupee-to-paise conversion.

## 7. AI assistance disclosure

ChatGPT was used for coding guidance, debugging assistance, analysis support, and documentation preparation. Training labels and test predictions were not generated by an external AI service. Test predictions were generated by the fine-tuned model using the specified inference process.
