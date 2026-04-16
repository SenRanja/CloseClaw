---
library_name: peft
license: other
base_model: /srv/scratch/z5526880/models/Qwen3-0.6B
tags:
- base_model:adapter:/srv/scratch/z5526880/models/Qwen3-0.6B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [/srv/scratch/z5526880/models/Qwen3-0.6B](https://huggingface.co//srv/scratch/z5526880/models/Qwen3-0.6B) on the sentiment_sft_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7091

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.9536        | 0.2075 | 50   | 0.9227          |
| 0.8282        | 0.4149 | 100  | 0.8208          |
| 0.7969        | 0.6224 | 150  | 0.7883          |
| 0.7652        | 0.8299 | 200  | 0.7626          |
| 0.6895        | 1.0373 | 250  | 0.7484          |
| 0.6818        | 1.2448 | 300  | 0.7334          |
| 0.6627        | 1.4523 | 350  | 0.7285          |
| 0.6646        | 1.6598 | 400  | 0.7182          |
| 0.6537        | 1.8672 | 450  | 0.7070          |
| 0.5667        | 2.0747 | 500  | 0.7122          |
| 0.5757        | 2.2822 | 550  | 0.7162          |
| 0.5557        | 2.4896 | 600  | 0.7112          |
| 0.5685        | 2.6971 | 650  | 0.7096          |
| 0.5727        | 2.9046 | 700  | 0.7088          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.11.0+cu130
- Datasets 4.0.0
- Tokenizers 0.22.2