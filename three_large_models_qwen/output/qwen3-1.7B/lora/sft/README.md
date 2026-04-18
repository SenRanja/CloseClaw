---
library_name: peft
license: other
base_model: /srv/scratch/z5526880/models/Qwen3-1.7B
tags:
- base_model:adapter:/srv/scratch/z5526880/models/Qwen3-1.7B
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

This model is a fine-tuned version of [/srv/scratch/z5526880/models/Qwen3-1.7B](https://huggingface.co//srv/scratch/z5526880/models/Qwen3-1.7B) on the sentiment_sft_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6046

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
| 0.8306        | 0.2075 | 50   | 0.7977          |
| 0.7036        | 0.4149 | 100  | 0.6887          |
| 0.6795        | 0.6224 | 150  | 0.6632          |
| 0.6577        | 0.8299 | 200  | 0.6394          |
| 0.5876        | 1.0373 | 250  | 0.6330          |
| 0.5877        | 1.2448 | 300  | 0.6205          |
| 0.5682        | 1.4523 | 350  | 0.6216          |
| 0.5731        | 1.6598 | 400  | 0.6096          |
| 0.5695        | 1.8672 | 450  | 0.6046          |
| 0.4903        | 2.0747 | 500  | 0.6094          |
| 0.4938        | 2.2822 | 550  | 0.6101          |
| 0.4832        | 2.4896 | 600  | 0.6070          |
| 0.4890        | 2.6971 | 650  | 0.6056          |
| 0.5012        | 2.9046 | 700  | 0.6048          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.11.0+cu130
- Datasets 4.0.0
- Tokenizers 0.22.2