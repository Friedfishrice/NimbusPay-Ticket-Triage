# NimbusPay Support Ticket Triage

Fine-tuning a small instruction model for structured NimbusPay support-ticket classification.

## Final model
- Base model: Qwen/Qwen2.5-1.5B-Instruct
- Fine-tuning: PEFT QLoRA, 4-bit NF4
- LoRA rank: 16
- LoRA alpha: 32
- Trainable parameters: 18,464,768 (1.182%)
- GPU: NVIDIA Tesla T4
- Maximum sequence length: 2304
- Training examples: 1,360

## Results
| Metric | Experiment 3 |
|---|---:|
| Development exact match | 81.0% |
| Mean field accuracy | 96.625% |
| JSON validity | 100% |
| Peak GPU memory | 9.415 GB |
| Training time | 33.35 minutes |

See report.md for the data audit, experiments, error analysis, limitations, and AI assistance disclosure.

## Repository structure
```text
NimbusPay_Ticket_Triage/
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
├── experiments/
├── predictions/
│   └── predictions_test.jsonl
├── cleaning_log.csv
├── notebook.ipynb
├── README.md
└── report.md
```

## Inference
Load Qwen/Qwen2.5-1.5B-Instruct, load the PEFT adapter from adapter/, apply the model chat template, and generate greedily with do_sample=False and max_new_tokens=200. Strip surrounding whitespace only.

The base model and original datasets are not included in this repository.

## Limitations
- The schema-rules prompt baseline was not completed.
- Raw-versus-cleaned and 16-bit LoRA-versus-QLoRA comparisons were not completed.
- Three fine-tuning runs were performed. Experiment 2 versus Experiment 3 is the controlled rank comparison.
- Experiment 1 used an earlier cleaned dataset and is not directly comparable.
- One of 400 test predictions contains the invalid model-generated date 2026-07-00. It was retained unchanged.

## AI assistance disclosure
ChatGPT was used for coding guidance, debugging, analysis support, and documentation preparation. Training and final inference were performed in the Colab environment.
