---
library_name: peft
license: other
base_model: /srv/scratch/z5526880/models/Qwen3-4B
tags:
- base_model:adapter:/srv/scratch/z5526880/models/Qwen3-4B
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

This model is a fine-tuned version of [/srv/scratch/z5526880/models/Qwen3-4B](https://huggingface.co//srv/scratch/z5526880/models/Qwen3-4B) on the sentiment_sft_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5539

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
| 0.7054        | 0.2075 | 50   | 0.6754          |
| 0.6115        | 0.4149 | 100  | 0.5981          |
| 0.5950        | 0.6224 | 150  | 0.5835          |
| 0.5743        | 0.8299 | 200  | 0.5668          |
| 0.5065        | 1.0373 | 250  | 0.5620          |
| 0.5057        | 1.2448 | 300  | 0.5533          |
| 0.4914        | 1.4523 | 350  | 0.5517          |
| 0.4953        | 1.6598 | 400  | 0.5480          |
| 0.4884        | 1.8672 | 450  | 0.5413          |
| 0.4028        | 2.0747 | 500  | 0.5595          |
| 0.4098        | 2.2822 | 550  | 0.5577          |
| 0.3910        | 2.4896 | 600  | 0.5554          |
| 0.4035        | 2.6971 | 650  | 0.5556          |
| 0.4106        | 2.9046 | 700  | 0.5541          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.11.0+cu130
- Datasets 4.0.0
- Tokenizers 0.22.2