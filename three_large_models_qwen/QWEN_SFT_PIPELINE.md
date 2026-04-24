# Qwen3 SFT Training Pipeline and Code Architecture

This document summarizes the SFT training process, data flow, code architecture, and output locations for Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B in the current code. The Qwen training in this project is a LoRA-supervised fine-tuning pipeline based on LLaMAFactory. The task is generative three-class sentiment analysis: the model reads movie reviews, generates brief explanations, and finally outputs `\boxed{positive}`, `\boxed{negative}`, or `\boxed{neutral}`.

## 1. total Pipeline

```text
annotated_reviews*.csv
        |
        v
scripts/prepare_sft_data.py
        |
        v
data/sft/sft_train.json
data/sft/sft_test.json
data/sft/val/source_0/sft_val.json
data/sft/val/source_1/sft_val.json
        |
        +-----------------------------+
        |                             |
        v                             v
Before-SFT evaluation          LLaMAFactory Qwen LoRA SFT
base Qwen model only           qwen_sft.yaml / scale-specific config
        |                             |
        v                             v
before_sft predictions         output/qwen3-*/lora/sft
                                      |
                                      v
                              After-SFT evaluation
                              base Qwen + LoRA adapter
                                      |
                                      v
                              after_sft predictions
```

The corresponding one-click entry is:

```bash
bash run_sft.sh
```

`run_sft.sh` performs four tasks:

1. It calls `scripts/prepare_sft_data.py` to generate SFT data.

2. It calls `scripts/evaluate_sft_val_predictions.py` to perform before-SFT validation set inference on the base model.

3. It calls `llamafactory-cli train qwen_sft.yaml` to perform LoRA SFT.

4. It calls `scripts/evaluate_sft_val_predictions.py` again, loads the LoRA adapter, and performs after-SFT validation set inference.

## 2. Code directory and responsibilities

| Path | Role |
| --- | --- |
| `run_sft.sh` | The main entry script for Qwen SFT, connecting data preparation, pre-training evaluation, LLaMAFactory training, and post-training evaluation. |
| `scripts/prepare_sft_data.py` | Reads `annotated_reviews*.csv` and constructs Alpaca-style SFT JSON data. |
| `configs/dataset_info.json` | Tells LLaMAFactory which JSON files `sentiment_sft_train` and `sentiment_sft_test` correspond to. |
| `configs/qwen_sft.yaml` | The Qwen SFT configuration file in the current root directory. |
| `6713group—final/configs/_qwen_sft_run.yaml` | The Qwen3-0.6B SFT configuration saved in the final directory. | | `6713group—final/configs/qwen_sft.yaml` | The Qwen3-4B SFT configuration saved in the final directory. |
| `scripts/evaluate_sft_val_predictions.py` | Performs generative inference on source-specific validation files and saves before/after CSVs. |
| `LlamaFactory/src/llamafactory/train/sft/workflow.py` | The main workflow for LLaMAFactory SFT training: loading the tokenizer, dataset, model, data collator, and trainer. |
| `LlamaFactory/src/llamafactory/train/sft/trainer.py` | LLaMAFactory's custom `CustomSeq2SeqTrainer`, which uses the default causal LM loss when no special loss is enabled. |
| `data/sft/` | SFT training, internal evaluation, and source-specific validation data. | | `output/qwen3-*/lora/sft` / `6713group—final/output/qwen3-*/lora/sft` | LoRA adapter, trainer state, loss curve, and training results. | | `eval_predictions*/` | CSV of Qwen's validation set predictions before and after SFT.

## 3. data preparation Pipeline

Data preparation entry point:

```bash
python scripts/prepare_sft_data.py
```

This script reads all files in the project's root directory:

```text
annotated_reviews*.csv
```

Main processing logic:

1. Read `review_text`, `sentiment`, `source`, `annotator_a`, and `annotator_b`.

2. Map the original tags to text tags:

```python
SENTIMENT_MAP = {
    "-1": "negative",
    "0": "neutral",
    "1": "positive",
}
```

3. If two annotators have inconsistent sentiments, exclude them from the SFT data. These disagreement samples are more suitable for subsequent DPO/preference-style data construction.

4. Convert each sample to Alpaca-style SFT format.

5. Stratify sampling for each source:

- validation: 500 samples per source

- test: 500 samples per source

- train: Remaining samples

6. Perform positive/negative downsampling on the train set to balance positive and negative samples, preserving neutral samples.

## 4. SFT Data Format

Each sample ultimately consists of four fields in JSON format:

```json
{
  "instruction": "Classify the sentiment of this movie review:\n\n<review text>",
  "input": "",
  "output": "<reason>\n\n\\boxed{positive}",
  "system": "You are a sentiment analysis assistant. Classify the sentiment..."
}
```

The model training objective is not simply to output a label, but to generate:

```text
reasoning / explanation

\boxed{label}
```
Therefore, Qwen SFT is essentially:

```text
Chat-style instruction

-> causal language modeling SFT

-> explanation + boxed sentiment label
```

## 5. Data Files and Quantity

The current data statistics in `data/sft/` are as follows:

| File | Total | Positive | Negative | Neutral | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `data/sft/sft_train.json` | 15,418 | 7,145 | 7,145 | 1,128 | LoRA SFT training |
| `data/sft/sft_test.json` | 1,000 | 608 | 331 | 61 | LLaMAFactory internal eval |
| `data/sft/val/source_0/sft_val.json` | 500 | 327 | 133 | 40 | source 0 before/after evaluation |
| `data/sft/val/source_1/sft_val.json` | 500 | 281 | 198 | 21 | source 1 before/after evaluation |

The `configs/dataset_info.json` file maps LLaMAFactory dataset names to these files:

```json
{
  "sentiment_sft_train": {
    "file_name": "../data/sft/sft_train.json"
  },
  "sentiment_sft_test": {
    "file_name": "../data/sft/sft_test.json"
  }
}
```

Used in training configuration:

```yaml
dataset_dir: ../configs
dataset: sentiment_sft_train
eval_dataset: sentiment_sft_test
```

## 6. SFT Configuration for Three Qwen Scales

The current root directory `configs/qwen_sft.yaml` is a runnable Qwen3-4B-Instruct configuration with a LoRA rank of 8:

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
finetuning_type: lora
lora_rank: 8
output_dir: ../output/qwen3-4b/lora/sft
```

The multi-scale Qwen SFT used for reporting results in the final experiments is primarily stored under `6713group-final/`. The final configuration uses LoRA rank 32. To ensure alignment between reporting and experimental results, it is recommended to retain the final configuration when finally committing the code, or to synchronize the root directory configuration with the final configuration.

| Model | Base model path recorded in artifacts | Config / Evidence | Output adapter dir | LoRA rank | Notes |
| --- | --- | --- | --- | ---: | --- |
| Qwen3-0.6B | `/srv/scratch/z5526880/models/Qwen3-0.6B` | `6713group—final/configs/_qwen_sft_run.yaml` | `6713group—final/output/qwen3-0.6B/lora/sft` | 32 | `save_steps=500`, `eval_steps=100`, eval batch size 32 |
| Qwen3-1.7B | `/srv/scratch/z5526880/models/Qwen3-1.7B` | `6713group—final/output/qwen3-1.7B/lora/sft/README.md` | `6713group—final/output/qwen3-1.7B/lora/sft` | 32 | README records same LR, batch size, seed, scheduler and epoch setup |
| Qwen3-4B | `/srv/scratch/z5526880/models/Qwen3-4B` | `6713group—final/configs/qwen_sft.yaml` | `6713group—final/output/qwen3-4B/lora/sft` | 32 | `save_steps=200`, `eval_steps=50`, eval batch size 64 |

Common SFT hyperparameters across the three final Qwen runs:

| Setting | Value |
| --- | --- |
| Stage | `sft` |
| Fine-tuning type | `lora` |
| LoRA target | `all` |
| Dataset | `sentiment_sft_train` |
| Eval dataset | `sentiment_sft_test` |
| Template | `qwen3_nothink` |
| Cutoff length | 512 |
| Per-device train batch size | 64 |
| Gradient accumulation steps | 1 |
| Learning rate | `1.0e-4` |
| Epochs | 3 |
| Scheduler | cosine |
| Warmup ratio | 0.1 |
| Precision | bf16 |
| Optimizer | AdamW fused, recorded in generated model card |
| Seed | 42, recorded in generated model card |

## 7. LLaMAFactory Internal training process

`run_sft.sh` The actual training command is:

```bash
cd configs
llamafactory-cli train qwen_sft.yaml
```

The general internal call chain of LLaMAFactory:

```text
llamafactory-cli train qwen_sft.yaml
        |
        v
llamafactory.cli:main
        |
        v
llamafactory.launcher.launch
        |
        v
llamafactory.train.tuner.run_exp
        |
        v
llamafactory.train.sft.workflow.run_sft
        |
        +--> load_tokenizer(model_args)
        +--> get_template_and_fix_tokenizer(tokenizer, data_args)
        +--> get_dataset(..., stage="sft")
        +--> load_model(tokenizer, model_args, finetuning_args, do_train)
        +--> SFTDataCollatorWith4DAttentionMask(...)
        +--> CustomSeq2SeqTrainer(...)
        +--> trainer.train()
        +--> trainer.save_model()
        +--> trainer.evaluate()
        +--> plot_loss(...)
```

Key documents:

- `LlamaFactory/src/llamafactory/train/sft/workflow.py`
- `LlamaFactory/src/llamafactory/train/sft/trainer.py`

### 7.1 Tokenizer / Template / Dataset

In `workflow.py`:

```python
tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]
template = get_template_and_fix_tokenizer(tokenizer, data_args)
dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
```

This completes the following steps:

1. Load the Qwen tokenizer.

2. Render the instruction-style sample using the `qwen3_nothink` template.

3. Convert `instruction/input/output/system` into Qwen chat-style input.

4. Construct the train/eval dataset.

### 7.2 Model / LoRA

在 `workflow.py` 中：

```python
model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
```

`finetuning_args` ：

```yaml
finetuning_type: lora
lora_rank: 32
lora_target: all
```

Therefore, the training is not full fine-tuning, but rather:

```text
base Qwen model mostly frozen
        +
trainable LoRA adapter matrices
```
After training, the adapter is saved, not the complete base model.

### 7.3 Data Collator & Loss Mask

`SFTDataCollatorWith4DAttentionMask` handles padding, attention masking, label padding, etc. In SFT, the prompt part is usually masked, while the assistant response part is used as the supervised target.

That is to say, the loss mainly affects:

``text

<explanation>

\boxed{label}

```

not the user prompt itself.

### 7.4 Loss Function

The following are not enabled in the current configuration:

```yaml

use_dft_loss

use_asft_loss

use_eaft_loss

```

Therefore, `CustomSeq2SeqTrainer.compute_loss` will use the default loss of `Seq2SeqTrainer` via HuggingFace. For causal language models, this is the token-level cross-entropy loss.

This can be understood as:

```text Predicting the next assistant token

vs
the actual assistant token

-> CrossEntropyLoss

```
Because the output includes the explanation and boxed label, the loss is trained not only on the final label but also on the explanation text.

## 8. Before / After SFT Evaluation Process

Evaluation Script:

```bash
python scripts/evaluate_sft_val_predictions.py

```
It reads:

```text
data/sft/val/source_0/sft_val.json
data/sft/val/source_1/sft_val.json

```
There are two evaluation methods:

### 8.1 Before SFT

Load only the base model, not the adapter:

```bash
python scripts/evaluate_sft_val_predictions.py \
--backend huggingface \
--base-model /path/to/Qwen3-4B \
--output-format csv \
--run-name before_sft

```
Purpose:

```text
Tests the raw performance of the base Qwen model under the current prompt and label schema.

` ... ```

### 8.2 After SFT

Load base model + LoRA adapter:

```bash python scripts/evaluate_sft_val_predictions.py \
--backend huggingface \
--base-model /path/to/Qwen3-4B \
--adapter output/qwen3-4B/lora/sft \
--output-format csv \
--run-name after_sft

```

Purpose:

```text Test the SFT adapter's adaptation to the project tagging system, output format, and neutral boundaries.

````` ```

### 8.3 Label Extraction

Parsing rules in `scripts/evaluate_sft_val_predictions.py`:

``python
BOXED_RE = re.compile(r"\\boxed\{(\w+)\}")
LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b")

```

Parsing priority:

1. Prioritize extraction from `\boxed{label}`.

2. If no boxed label exists, search for the last occurrence of `positive/negative/neutral` in the text.

3. If none are found, retain the original output; this is typically considered an invalid/wrong prediction during evaluation.
## 9. Training Outputs

Core outputs of each Qwen SFT run:

| File / Directory | Meaning |

| --- | --- |

| `adapter_model.safetensors` | LoRA adapter weights |

| `adapter_config.json` | LoRA adapter configuration |

| `trainer_state.json` | trainer step, loss, runtime, log history |

| `trainer_log.jsonl` | training logs |

| `all_results.json` | train/eval summary |

| `eval_results.json` | eval loss/runtime summary |

| `training_eval_loss.png` | loss graph |

| `README.md` | LLaMAFactory automatically generated adapter model card |

Main outputs in the final directory adapter:

```text 6713group—final/output/qwen3-0.6B/lora/sft
6713group—final/output/qwen3-1.7B/lora/sft
6713group—final/output/qwen3-4B/lora/sft
```

## 10. Training Results at Three Qwen Scales

These results are from `all_results.json` in the final directory.

| Model | Epoch | Train Loss | Eval Loss | Train Runtime | Train Samples/s | Eval Runtime | Eval Samples/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 3.0 | 0.7417 | 0.7091 | 563.5s | 82.082 | 3.667s | 272.672 |
| Qwen3-1.7B | 3.0 | 0.6786 | 0.6046 | 857.8s | 53.924 | 5.534s | 180.715 |
| Qwen3-4B | 3.0 | 0.5703 | 0.5539 | 1681.3s | 27.510 | 10.485s | 95.371 |

As we can see, as the model size increases from 0.6B to 4B:

- Train loss monotonically decreases;

Evaluation loss monotonically decreases;

- Training time increases significantly;

- Samples/s decreases significantly;

- Larger models are more expensive, but offer better fitting and generalization metrics.

## 11. Validation Set Performance at Three Qwen Scales

The raw validation metrics recalculated from the current CSV are as follows. Here, we use the average of the metrics calculated separately for source0 and source1.

| Model | Mean Macro-F1 | Mean Accuracy | Mean Weighted-F1 | Mean Macro-P | Mean Macro-R | Macro-F1 Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B before SFT | 0.5098 | 0.6610 | 0.6759 | 0.5136 | 0.5117 | 0.0149 |
| Qwen3-0.6B after SFT | 0.8358 | 0.9390 | 0.9380 | 0.8472 | 0.8280 | 0.0315 |
| Qwen3-1.7B before SFT | 0.4173 | 0.5620 | 0.5809 | 0.4315 | 0.4342 | 0.0418 |

| Qwen3-1.7B after SFT | 0.8512 | 0.9440 | 0.9464 | 0.8349 | 0.8730 | 0.0351 |

| Qwen3-4B before SFT | 0.5098 | 0.6760 | 0.6834 | 0.5164 | 0.5275 | 0.0799 |

| Qwen3-4B after SFT | 0.8687 | 0.9490 | 0.9536 | 0.8412 | 0.9206 | 0.0005 |

Conclusion:

1. All three Qwen models show significant improvement after SFT.

2. Qwen3-4B after SFT has the highest mean Macro-F1 score.

3. Qwen3-4B has the smallest source gap after SFT, indicating the most stable performance across sources.

4. Although Qwen3-0.6B has the smallest source gap, its performance after SFT is already very strong, making it suitable for rapid experimentation or resource-constrained scenarios.

## 12. Recommended Operation Method for Training Qwen at Different Scales

Since `run_sft.sh` currently calls the following fixed command:

```bash
cd configs

llamafactory-cli train qwen_sft.yaml

```

When training at different scales, the actual operation is usually to modify or copy `qwen_sft.yaml`:

```text

configs/qwen_sft_0.6b.yaml

configs/qwen_sft_1.7b.yaml

configs/qwen_sft_4b.yaml

```

Each file only needs to be modified as follows:

```yaml

model_name_or_path: /path/to/Qwen3-0.6B

output_dir: ../output/qwen3-0.6B/lora/sft

```

Or:

```yaml

model_name_or_path: /path/to/Qwen3-1.7B
output_dir: ../output/qwen3-1.7B/lora/sft

```

Or:

```yaml
model_name_or_path: /path/to/Qwen3-4B
output_dir: ../output/qwen3-4B/lora/sft

```

Then run the following commands respectively:

```bash
cd configs
llamafactory-cli train qwen_sft_0.6b.yaml
llamafactory-cli train qwen_sft_1.7b.yaml
llamafactory-cli train qwen_sft_4b.yaml

```

If you continue using `run_sft.sh`, you need to modify the following accordingly:

```bash
BASE_MODEL=...
SFT_ADAPTER=...

```

And In `configs/qwen_sft.yaml`:

```yaml
model_name_or_path
output_dir

```
## 13. Alignment Issues to Note in the Current Code

There are two sets of configuration traces in the current project:

1. Current configuration in the root directory:

```text
configs/qwen_sft.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
lora_rank: 8
output_dir: ../output/qwen3-4b/lora/sft

```
2. Configuration corresponding to the final result:

```text
6713group—final/configs/qwen_sft.yaml
model_name_or_path: /srv/scratch/z5526880/models/Qwen3-4B
lora_rank: 32
output_dir: ../output/qwen3-4B/lora/sft


The Qwen scaling results in the report correspond to the final configuration and final output. To avoid confusion during submission, it is recommended to copy the three final Qwen SFT configurations to the root directory `configs/`, for example:

```text
configs/qwen_sft_0.6b.yaml
configs/qwen_sft_1.7b.yaml
configs/qwen_sft_4b.yaml

```

This will ensure complete alignment of the report, code, checkpoint, and execution commands.

## 14. One-sentence summary

The current Qwen SFT pipeline is:

```text
annotated CSV

-> instruction-style SFT JSON

-> Qwen base model before-SFT evaluation

-> LLaMAFactory LoRA SFT

-> adapter checkpoint

-> base model + adapter after-SFT evaluation

-> cross-source and scaling analysis

```

The 0.6B, 1.7B, and 4B scales share the same data format, training objectives, and evaluation methods; the main differences lie in the base model size, training time, loss, validation set performance, and cross-source stability.