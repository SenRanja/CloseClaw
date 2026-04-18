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

LLaMAFactory 内部大致调用链：

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

关键文件：

- `LlamaFactory/src/llamafactory/train/sft/workflow.py`
- `LlamaFactory/src/llamafactory/train/sft/trainer.py`

### 7.1 Tokenizer / Template / Dataset

在 `workflow.py` 中：

```python
tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]
template = get_template_and_fix_tokenizer(tokenizer, data_args)
dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
```

这里完成：

1. 加载 Qwen tokenizer。
2. 使用 `qwen3_nothink` 模板渲染 instruction-style sample。
3. 将 `instruction/input/output/system` 转成 Qwen chat-style 输入。
4. 构造 train/eval dataset。

### 7.2 Model / LoRA

在 `workflow.py` 中：

```python
model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
```

`finetuning_args` 里指定：

```yaml
finetuning_type: lora
lora_rank: 32
lora_target: all
```

因此训练时不是 full fine-tuning，而是：

```text
base Qwen model mostly frozen
        +
trainable LoRA adapter matrices
```

训练完成后保存的是 adapter，而不是完整 base model。

### 7.3 Data Collator 和 Loss Mask

`SFTDataCollatorWith4DAttentionMask` 会负责 padding、attention mask、label padding 等。SFT 中 prompt 部分通常被 mask 掉，assistant response 部分才作为 supervised target。

也就是说 loss 主要作用在：

```text
<explanation>

\boxed{label}
```

而不是用户 prompt 本身。

### 7.4 Loss Function

当前配置没有启用：

```yaml
use_dft_loss
use_asft_loss
use_eaft_loss
```

因此 `CustomSeq2SeqTrainer.compute_loss` 会走 HuggingFace `Seq2SeqTrainer` 默认 loss。对于 causal language model，这就是 token-level cross entropy loss。

可以理解为：

```text
预测下一个 assistant token
        vs
真实 assistant token
        -> CrossEntropyLoss
```

因为 output 包含 explanation 和 boxed label，所以 loss 不只训练最终 label，也训练解释文本。

## 8. Before / After SFT 评估流程

评估脚本：

```bash
python scripts/evaluate_sft_val_predictions.py
```

它读取：

```text
data/sft/val/source_0/sft_val.json
data/sft/val/source_1/sft_val.json
```

评估分两种：

### 8.1 Before SFT

只加载 base model，不加载 adapter：

```bash
python scripts/evaluate_sft_val_predictions.py \
  --backend huggingface \
  --base-model /path/to/Qwen3-4B \
  --output-format csv \
  --run-name before_sft
```

用途：

```text
测试 base Qwen 在当前 prompt 和 label schema 下的原始表现。
```

### 8.2 After SFT

加载 base model + LoRA adapter：

```bash
python scripts/evaluate_sft_val_predictions.py \
  --backend huggingface \
  --base-model /path/to/Qwen3-4B \
  --adapter output/qwen3-4B/lora/sft \
  --output-format csv \
  --run-name after_sft
```

用途：

```text
测试 SFT adapter 对项目标签体系、输出格式和 neutral 边界的适配效果。
```

### 8.3 Label Extraction

`scripts/evaluate_sft_val_predictions.py` 中的解析规则：

```python
BOXED_RE = re.compile(r"\\boxed\{(\w+)\}")
LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b")
```

解析优先级：

1. 优先从 `\boxed{label}` 中提取。
2. 如果没有 boxed label，则从文本中找最后出现的 `positive/negative/neutral`。
3. 如果都找不到，则保留原始输出，评估时通常视为 invalid prediction / wrong prediction。

## 9. 训练产物

每个 Qwen SFT run 的核心产物：

| File / Directory | Meaning |
| --- | --- |
| `adapter_model.safetensors` | LoRA adapter 权重 |
| `adapter_config.json` | LoRA adapter 配置 |
| `trainer_state.json` | trainer step、loss、runtime、log history |
| `trainer_log.jsonl` | 训练日志 |
| `all_results.json` | train/eval summary |
| `eval_results.json` | eval loss/runtime summary |
| `training_eval_loss.png` | loss 曲线图 |
| `README.md` | LLaMAFactory 自动生成的 adapter model card |

final 目录中的主要 adapter：

```text
6713group—final/output/qwen3-0.6B/lora/sft
6713group—final/output/qwen3-1.7B/lora/sft
6713group—final/output/qwen3-4B/lora/sft
```

## 10. 三个 Qwen 尺度的训练结果

这些结果来自 final 目录中的 `all_results.json`。

| Model | Epoch | Train Loss | Eval Loss | Train Runtime | Train Samples/s | Eval Runtime | Eval Samples/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 3.0 | 0.7417 | 0.7091 | 563.5s | 82.082 | 3.667s | 272.672 |
| Qwen3-1.7B | 3.0 | 0.6786 | 0.6046 | 857.8s | 53.924 | 5.534s | 180.715 |
| Qwen3-4B | 3.0 | 0.5703 | 0.5539 | 1681.3s | 27.510 | 10.485s | 95.371 |

可以看到，随着模型从 0.6B 增加到 4B：

- train loss 单调下降；
- eval loss 单调下降；
- 训练时间明显增加；
- samples/s 明显下降；
- 大模型更贵，但拟合和泛化指标更好。

## 11. 三个 Qwen 尺度的验证集表现

当前 CSV 重新计算出的 raw validation metrics 如下。这里使用 source0/source1 分别计算后取平均的口径。

| Model | Mean Macro-F1 | Mean Accuracy | Mean Weighted-F1 | Mean Macro-P | Mean Macro-R | Macro-F1 Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B before SFT | 0.5098 | 0.6610 | 0.6759 | 0.5136 | 0.5117 | 0.0149 |
| Qwen3-0.6B after SFT | 0.8358 | 0.9390 | 0.9380 | 0.8472 | 0.8280 | 0.0315 |
| Qwen3-1.7B before SFT | 0.4173 | 0.5620 | 0.5809 | 0.4315 | 0.4342 | 0.0418 |
| Qwen3-1.7B after SFT | 0.8512 | 0.9440 | 0.9464 | 0.8349 | 0.8730 | 0.0351 |
| Qwen3-4B before SFT | 0.5098 | 0.6760 | 0.6834 | 0.5164 | 0.5275 | 0.0799 |
| Qwen3-4B after SFT | 0.8687 | 0.9490 | 0.9536 | 0.8412 | 0.9206 | 0.0005 |

结论：

1. 三个 Qwen 模型 SFT 后都有明显提升。
2. Qwen3-4B after SFT 的 mean Macro-F1 最高。
3. Qwen3-4B after SFT 的 source gap 最小，说明跨 source 最稳定。
4. Qwen3-0.6B 虽然最小，但 SFT 后表现已经很强，适合快速实验或资源受限场景。

## 12. 训练不同尺度 Qwen 的推荐操作方式

由于 `run_sft.sh` 当前固定调用：

```bash
cd configs
llamafactory-cli train qwen_sft.yaml
```

训练不同尺度时，实际操作一般是修改或复制 `qwen_sft.yaml`：

```text
configs/qwen_sft_0.6b.yaml
configs/qwen_sft_1.7b.yaml
configs/qwen_sft_4b.yaml
```

每个文件只需要主要修改：

```yaml
model_name_or_path: /path/to/Qwen3-0.6B
output_dir: ../output/qwen3-0.6B/lora/sft
```

或：

```yaml
model_name_or_path: /path/to/Qwen3-1.7B
output_dir: ../output/qwen3-1.7B/lora/sft
```

或：

```yaml
model_name_or_path: /path/to/Qwen3-4B
output_dir: ../output/qwen3-4B/lora/sft
```

然后分别运行：

```bash
cd configs
llamafactory-cli train qwen_sft_0.6b.yaml
llamafactory-cli train qwen_sft_1.7b.yaml
llamafactory-cli train qwen_sft_4b.yaml
```

如果继续使用 `run_sft.sh`，则需要同步修改：

```bash
BASE_MODEL=...
SFT_ADAPTER=...
```

以及 `configs/qwen_sft.yaml` 中的：

```yaml
model_name_or_path
output_dir
```

## 13. 当前代码需要注意的对齐问题

当前项目中存在两套配置痕迹：

1. 根目录当前配置：

```text
configs/qwen_sft.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
lora_rank: 8
output_dir: ../output/qwen3-4b/lora/sft
```

2. final 结果对应配置：

```text
6713group—final/configs/qwen_sft.yaml
model_name_or_path: /srv/scratch/z5526880/models/Qwen3-4B
lora_rank: 32
output_dir: ../output/qwen3-4B/lora/sft
```

报告中的 Qwen scaling 结果对应 final 配置和 final output。为了提交时避免混淆，建议将最终使用的三份 Qwen SFT 配置复制到根目录 `configs/` 下，例如：

```text
configs/qwen_sft_0.6b.yaml
configs/qwen_sft_1.7b.yaml
configs/qwen_sft_4b.yaml
```

这样 report、code、checkpoint 和运行命令就能完全对齐。

## 14. 一句话总结

当前 Qwen SFT pipeline 是：

```text
annotated CSV
 -> instruction-style SFT JSON
 -> Qwen base model before-SFT evaluation
 -> LLaMAFactory LoRA SFT
 -> adapter checkpoint
 -> base model + adapter after-SFT evaluation
 -> cross-source and scaling analysis
```

其中 0.6B、1.7B、4B 三个尺度共享同一套数据格式、训练目标和评估方式；区别主要在 base model size、训练耗时、loss、验证集表现和跨 source 稳定性。
