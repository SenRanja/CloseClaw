# COMP6713 情感分析项目：建模、微调与偏好优化 Pipeline 计划书

## 1. 项目背景与目标 (Introduction & Objectives)

在数据集构造阶段完成后，本项目的重点将从“如何得到高质量、难度分级且防泄漏的数据”转向“如何基于该数据建立一个可复现、可比较、可落地部署的建模 pipeline”。本阶段不仅要训练能够完成电影评论情感分析任务的模型，还要通过严格的对照实验回答三个核心问题：

第一，传统稀疏特征模型在当前数据集上的能力上限在哪里；第二，标准编码器模型如 BERT-base 在困难样本上的提升幅度有多大；第三，小型生成式语言模型在监督微调和偏好优化之后，是否能够在复杂语境、结构化输出和解释一致性方面进一步优于前两者。

因此，本计划书将建模阶段划分为四条连续路线：

- `TF-IDF + Linear SVM` 稀疏基线。
- `BERT-base` 监督微调。
- 小型 `Qwen` 监督微调 (SFT)。
- 基于分歧样本偏好对的 `Qwen-DPO` 增强。

与前一份数据工程文档相对应，本文件强调的不是“收什么数据”，而是“每种模型吃什么输入、如何训练、以什么标准比较、最终输出什么资产”。这样可以保证建模阶段从一开始就服务于最终报告、展示和命令行 demo，而不是只停留在零散训练脚本层面。

## 2. 建模输入、任务边界与数据契约 (Modeling Inputs, Task Scope & Data Contract)

### 2.1 输入数据文件与用途

建模阶段假定数据工程阶段已经产出一组固定且可复用的数据文件。推荐直接使用以下标准文件作为统一输入：

- `train_balanced.jsonl`
- `val_balanced.jsonl`
- `test_balanced.jsonl`
- `val_natural.jsonl`
- `test_natural.jsonl`
- `annotated_reviews.jsonl`
- `pairwise_dpo.jsonl`

其中，`train_balanced.jsonl`、`val_balanced.jsonl` 和 `test_balanced.jsonl` 主要服务于受控模型比较；`val_natural.jsonl` 与 `test_natural.jsonl` 主要服务于真实分布评测；`annotated_reviews.jsonl` 用于监督微调、误差分析和多视角统计；`pairwise_dpo.jsonl` 则专门用于 Qwen 的 DPO 偏好优化阶段。

为了保证不同模型之间的数据接口稳定，监督学习输入记录建议至少包含以下字段：

- `id`
- `text_clean`
- `review_text`
- `sentiment`
- `difficulty`
- `ambiguous_flag`
- `source`
- `source_type`
- `movie_title`
- `verified_by`

对于 Qwen 的 SFT/DPO 阶段，还建议保留可生成结构化输出所需的字段，例如 `judge_analysis`、`annotator_a` 与 `annotator_b`，以便从最终审定结果中构造监督目标或偏好对。若上游 annotation log 中同时保留了 `confidence`，也仅将其视为审计元数据或样本筛选参考，而**不**将其纳入下游 SFT/DPO 的目标输出。

### 2.1.1 文件之间的整体关系

上述 7 个文件并不是彼此孤立的，而是一个“主表 + 派生子集”的结构。`annotated_reviews.jsonl` 是整个建模阶段最完整的主数据表，保存所有已经完成最终裁决的样本；`train_balanced.jsonl`、`val_balanced.jsonl`、`test_balanced.jsonl`、`val_natural.jsonl` 与 `test_natural.jsonl` 都是从这张主表按照固定 split 和不同采样策略派生出来的监督学习文件；`pairwise_dpo.jsonl` 则进一步从训练来源的分歧样本中提取 `prompt/chosen/rejected` 结构，专门服务于 Qwen 的 DPO 偏好优化阶段。

换言之，建模阶段的数据流可以抽象为：

`annotated_reviews.jsonl` -> `train/val/test` 监督文件 -> `pairwise_dpo.jsonl`

为了避免误解，下面所有 JSON case 都是**推荐 schema 示例**，用于说明每个文件应该长什么样，而不是代表当前目录中已经真实存在的最终数据文件。

### 2.1.2 `annotated_reviews.jsonl`：最终标注主表

`annotated_reviews.jsonl` 是最完整的“主数据表”。它的核心作用不是只给某一个模型训练，而是承担整个建模阶段的数据中台角色。它既可以用于构造 BERT/Qwen 的监督微调样本，也可以用于误差分析、来源统计、难度统计、人工抽检以及后续派生 balanced/natural 数据集。

这张表建议保留四类信息：

- 原始文本字段：`review_text`、`text_clean`
- 最终标签字段：`sentiment`、`difficulty`、`ambiguous_flag`
- 来源与切分字段：`source`、`source_type`、`movie_title`、`split`
- 审计与仲裁字段：`verified_by`、`annotator_a`、`annotator_b`、`judge_analysis`

推荐 case 如下：

```json
{
  "id": "sample_0002",
  "split": "train",
  "source": "Reddit",
  "source_type": "unlabeled_crawl",
  "movie_title": "Dune: Part Two",
  "review_text": "Visually overwhelming in the best way, but I still cannot tell whether I loved the movie or just admired the production design.",
  "text_clean": "Visually overwhelming in the best way, but I still cannot tell whether I loved the movie or just admired the production design.",
  "sentiment": "negative",
  "difficulty": 2,
  "ambiguous_flag": false,
  "verified_by": "supreme_judge",
  "annotator_a": {
    "sentiment": "positive",
    "difficulty": 2
  },
  "annotator_b": {
    "sentiment": "negative",
    "difficulty": 2
  },
  "judge_analysis": "Surface praise exists, but the dominant sentiment is negative."
}
```

### 2.1.3 `train_balanced.jsonl`：平衡训练集

`train_balanced.jsonl` 是传统监督模型最直接使用的主训练文件。它来源于 `annotated_reviews.jsonl`，但只保留训练所需的核心字段，并对类别分布做平衡化处理。这里的“平衡”主要指两件事：第一，正负样本比例尽量接近 `1:1`；第二，`difficulty = 1/2/3` 的样本规模尽量接近，从而让模型在训练时不会被某一类简单样本压倒性主导。

这个文件主要服务于：

- `TF-IDF + Linear SVM`
- `BERT-base`
- Qwen 的部分监督微调实验

推荐 case 如下：

```json
{
  "id": "train_0142",
  "split": "train",
  "text_clean": "A beautifully acted film that loses some momentum in the middle but remains emotionally rewarding.",
  "review_text": "A beautifully acted film that loses some momentum in the middle but remains emotionally rewarding.",
  "sentiment": "positive",
  "difficulty": 2,
  "ambiguous_flag": false,
  "source": "IMDb",
  "source_type": "public_dataset",
  "movie_title": "Past Lives",
  "verified_by": "strong_consensus"
}
```

### 2.1.4 `val_balanced.jsonl`：平衡验证集

`val_balanced.jsonl` 是受控验证集，主要用于超参数选择、checkpoint 选择和模型之间的公平横向比较。它和 `train_balanced.jsonl` 的字段设计应该尽量保持一致，这样模型代码只需要共用一套 dataset loader。它与训练集的本质区别不在 schema，而在用途：它绝不能参与训练，也不应该被反复当成最终汇报结果使用。

推荐 case 如下：

```json
{
  "id": "val_0031",
  "split": "val",
  "text_clean": "The performances are committed, but the script never finds a convincing emotional center.",
  "review_text": "The performances are committed, but the script never finds a convincing emotional center.",
  "sentiment": "negative",
  "difficulty": 2,
  "ambiguous_flag": false,
  "source": "RottenTomatoes",
  "source_type": "public_dataset",
  "movie_title": "Maestro",
  "verified_by": "weak_consensus_upgraded"
}
```

### 2.1.5 `test_balanced.jsonl`：平衡测试集

`test_balanced.jsonl` 是受控测试集，用于最终报告中最核心的“公平对比结果”。它适合回答“在相同、受控、平衡的数据条件下，SVM、BERT、Qwen-SFT 和 Qwen-DPO 到底谁更强”。由于这个文件承担最终横向比较功能，因此在整个调参期间不应被频繁查看，也不应被任何训练脚本间接利用。

推荐 case 如下：

```json
{
  "id": "test_0088",
  "split": "test",
  "text_clean": "It is technically impressive, yet the whole experience feels strangely empty.",
  "review_text": "It is technically impressive, yet the whole experience feels strangely empty.",
  "sentiment": "negative",
  "difficulty": 2,
  "ambiguous_flag": false,
  "source": "Letterboxd",
  "source_type": "unlabeled_crawl",
  "movie_title": "Blade Runner 2049",
  "verified_by": "supreme_judge"
}
```

### 2.1.6 `val_natural.jsonl` 与 `test_natural.jsonl`：自然分布验证/测试集

这两个文件和 balanced 版本最大的区别，不在字段设计，而在**样本分布策略**。它们保留更接近真实平台评论流的自然比例，不对正负类、难度层和平台来源做强制拉平。因此，它们更适合回答“模型在真实环境里是否仍然可靠”，而不是“模型在受控条件下谁更强”。

`val_natural.jsonl` 主要用于：

- 观察模型是否对自然分布更敏感
- 分析是否存在平台偏差
- 辅助选择更稳健的模型版本

推荐 case 如下：

```json
{
  "id": "val_nat_0017",
  "split": "val",
  "text_clean": "I wanted to like it more than I did. Some scenes are excellent, but the pacing keeps ruining the mood.",
  "review_text": "I wanted to like it more than I did. Some scenes are excellent, but the pacing keeps ruining the mood.",
  "sentiment": "negative",
  "difficulty": 2,
  "ambiguous_flag": false,
  "source": "IMDb",
  "source_type": "public_dataset",
  "movie_title": "The Last Duel",
  "verified_by": "strong_consensus"
}
```

`test_natural.jsonl` 则是最接近“真实部署效果”的最终评测文件。它比 `test_balanced.jsonl` 更能反映模型在真实数据比例下的实际可用性，特别适合放在 report 和 presentation 的“现实场景结果”部分。

推荐 case 如下：

```json
{
  "id": "test_nat_0044",
  "split": "test",
  "text_clean": "Not exactly bad, just never as clever or moving as it thinks it is.",
  "review_text": "Not exactly bad, just never as clever or moving as it thinks it is.",
  "sentiment": "negative",
  "difficulty": 3,
  "ambiguous_flag": false,
  "source": "Reddit",
  "source_type": "unlabeled_crawl",
  "movie_title": "Saltburn",
  "verified_by": "supreme_judge"
}
```

### 2.1.7 `pairwise_dpo.jsonl`：DPO 偏好对训练集

`pairwise_dpo.jsonl` 是这 7 个文件里最特殊的一类，因为它不再是“单条文本 + 单个标签”的普通监督数据，而是面向偏好优化的 pairwise 数据。它通常来自 `annotated_reviews.jsonl` 中的分歧样本，尤其是两个模型在 `sentiment` 上直接冲突的 `Zero Consensus` 记录。核心目标不是再教模型一次标签定义，而是告诉模型：面对同一条输入时，什么样的结构化答案更值得偏好。

这个文件的核心字段建议为：

- `id`
- `source_id`
- `pair_type`
- `prompt`
- `chosen`
- `rejected`
- `verified_by`
- `final_label`

其中，`prompt` 是任务指令加评论文本，`chosen` 与 `rejected` 都是完整 JSON 字符串。需要特别强调的是：虽然上游自动标注阶段可能保留 `confidence` 字段，但它**不应进入**这里的 `chosen/rejected` 目标字符串。这样 DPO 学到的是“更优的标签与解释模式”，而不是模仿一个未必校准良好的置信度数字。

推荐 case 如下：

```json
{
  "id": "sample_0002:sentiment_conflict:annotator_b_over_annotator_a",
  "source_id": "sample_0002",
  "pair_type": "sentiment_conflict",
  "verified_by": "supreme_judge",
  "prompt": "You are given a movie review. Return only valid JSON with the fields `sentiment`, `difficulty`, `ambiguous_flag`, and `reasoning`.\n\nReview:\nVisually overwhelming in the best way, but I still cannot tell whether I loved the movie or just admired the production design.",
  "chosen": "{\"sentiment\":\"negative\",\"difficulty\":2,\"ambiguous_flag\":false,\"reasoning\":\"Admiration does not translate into genuine enjoyment.\"}",
  "rejected": "{\"sentiment\":\"positive\",\"difficulty\":2,\"ambiguous_flag\":true,\"reasoning\":\"The reviewer sounds impressed overall.\"}",
  "final_label": {
    "sentiment": "negative",
    "difficulty": 2,
    "ambiguous_flag": false
  }
}
```

### 2.1.8 建议统一保留的核心字段

为了让 SVM、BERT、Qwen-SFT、Qwen-DPO 和误差分析脚本尽量共用同一套读取逻辑，普通监督文件建议统一保留以下核心字段：

- `id`
- `split`
- `text_clean`
- `review_text`
- `sentiment`
- `difficulty`
- `ambiguous_flag`
- `source`
- `source_type`
- `movie_title`
- `verified_by`

只要这组字段稳定存在，后续不论是做分类训练、生成式监督微调、分层统计还是错误分析，代码层的耦合度都会显著降低。

### 2.2 主任务与辅助任务定义

本阶段的主任务仍然是**二分类电影评论情感分析**，即预测每条评论的主导情感为 `positive` 或 `negative`。因此，所有模型的主评测指标都必须围绕情感预测展开，而不能被难度分级、解释文本或 JSON 生成质量完全替代。

与此同时，本项目还保留三个辅助目标：

- 使用 `difficulty` 作为分层评测维度，验证难度标签是否反映真实建模难度。
- 使用 `ambiguous_flag` 识别高风险样本，并分析模型在边界样本上的鲁棒性。
- 对生成式模型要求结构化输出，以支持后续 CLI 和 demo 中的可解释展示。

换言之，`difficulty` 与 `ambiguous_flag` 是重要分析标签，但不取代主任务；`reasoning` 是生成式模型的重要输出字段。若上游保留 `confidence`，它也只作为辅助分析元数据存在，而不作为任何下游训练目标。

### 2.3 数据读取与样本过滤方法

在建模阶段，推荐统一使用 `text_clean` 作为传统模型和 BERT 的默认输入文本，原因是该字段已经完成基础清洗，能够减少 HTML、URL 和异常噪声对稀疏特征与 tokenizer 的干扰。对于 Qwen 这类生成式模型，建议优先使用语义更自然的 `review_text`，若该字段缺失，再回退到 `text_clean`。

关于 `ambiguous_flag = true` 的样本，本项目建议采用“双轨策略”。主实验保留所有已经过最终裁决的样本，以保证数据分布完整；同时额外构造一个高纯度训练子集，排除 `ambiguous_flag = true` 的样本，用作补充实验。这样既能保持主结果与真实任务一致，也能观察标签纯度提升后模型表现是否更加稳定。

### 2.4 数据切分与泄漏控制方法

建模阶段不应重新随机切分数据，而应**严格复用数据工程阶段已经固定的 split 清单**。任何 SVM、BERT、Qwen 或 DPO 相关实验，都必须建立在同一组 `train/val/test` 划分之上。否则，模型之间的比较将失去公平性，报告中的结果也无法被复现。

对于 `pairwise_dpo.jsonl`，同样要遵守原始样本所属 split 的边界。更稳妥的做法是仅从训练集来源的分歧样本构造 DPO 训练对，而不要把验证集或测试集中的 pairwise 记录混入 DPO 训练阶段。否则，即使没有显式看到测试标签，模型也可能通过偏好学习间接接触测试分布信息。

### 2.5 本阶段输出

建模阶段的输出不应只是一组模型权重，而应是一整套可审计、可比较、可展示的实验资产。推荐最终产出至少包括：

- `svm_vectorizer.joblib`
- `svm_model.joblib`
- `bert_best_checkpoint/`
- `qwen_sft_adapter/`
- `qwen_dpo_adapter/`
- `eval_balanced.json`
- `eval_natural.json`
- `error_analysis_cases.jsonl`
- `prediction_samples.jsonl`

这些文件将共同支撑后续报告撰写、命令行测试和 Gradio demo，而不是让模型训练结果散落在临时 notebook 中。

## 3. 总体建模路线与比较框架 (Overall Modeling Roadmap & Comparison Design)

### 3.1 横向比较方法

本项目的横向比较分为三类模型家族：

- **传统稀疏模型**：`TF-IDF + Linear SVM`
- **编码器判别模型**：`BERT-base`
- **解码器生成模型**：小型 `Qwen`

这种横向比较有两个意义。第一，它能够展示“从词袋特征到上下文编码，再到生成式语言模型”的能力跃迁。第二，它能帮助我们判断困难样本到底需要多强的建模能力，还是仅靠更好的特征与更稳的训练就已经足够。

### 3.2 纵向比较方法

除横向比较外，本项目还会在小型 Qwen 上进行一组纵向比较：

- `Qwen-SFT`
- `Qwen-DPO`

这组比较的目的不是证明“DPO 一定优于 SFT”，而是验证：在已经具备基本任务能力之后，基于分歧样本构造的偏好对是否能够进一步改善模型在冲突样本、边界样本和结构化输出稳定性方面的表现。因此，Qwen 的实验不应只给出一个最终最好分数，而应呈现一个从普通监督学习到偏好优化的完整演化过程。

## 4. 稀疏基线模型：TF-IDF + Linear SVM (Sparse Baseline Pipeline)

### 4.1 选择 Linear SVM 作为主基线的原因

本项目将 `TF-IDF + Linear SVM` 作为建模阶段的主基线，而不是把它仅仅视为“随便跑一下的参考线”。原因在于，SVM 在中小规模文本分类任务中通常比简单线性回归类模型更稳，对高维稀疏特征也较为友好。若它在 Level 1/2/3 上已经表现出清晰、稳定的难度下降趋势，那么后续更复杂模型的提升就更有解释价值。

同时，Linear SVM 的决策边界清晰、训练成本低、调参空间适中，非常适合作为本项目所有复杂实验的比较锚点。后续 BERT 和 Qwen 的提升，必须建立在“明显超过这个稀疏强基线”的前提上，才更具说服力。

### 4.2 特征工程与训练方法

SVM 的输入文本推荐使用 `text_clean`。特征提取层可采用 `TF-IDF` 的 unigram 与 bigram 组合，并结合 `min_df`、`max_df` 和 `max_features` 控制噪声词与超高频词带来的污染。一个较稳妥的初始设置是：

- `ngram_range = (1, 2)`
- `min_df = 3`
- `max_df = 0.95`
- `sublinear_tf = True`
- `max_features = 30000` 或 `50000`

训练目标直接使用二分类标签 `sentiment`。对于类别已经平衡的训练集，可先使用默认类别权重；若后续加入自然分布训练实验，则可补充 `class_weight = balanced` 作为对照设置。

### 4.3 超参数搜索与模型选择方法

SVM 的主要搜索空间应聚焦于 `C`、特征维度和 n-gram 范围，而不是无限扩大网格。建议先在以下范围内进行小规模搜索：

- `C in {0.1, 1.0, 5.0}`
- `max_features in {30000, 50000}`
- `ngram_range in {(1, 1), (1, 2)}`

模型选择标准不应只看 `Accuracy`，而应以 `val_balanced` 上的 `Macro-F1` 作为主选择指标。因为本项目后续还要报告不同难度层上的表现，单一准确率很容易掩盖边界样本和少数模式上的缺陷。

### 4.4 SVM 推理输出与分析接口

尽管 SVM 本身只输出分类结果和决策分数，本项目仍建议为其统一封装推理接口，使其能够输出类似下面的结构：

```json
{
  "id": "sample_0042",
  "model_name": "tfidf_linear_svm",
  "sentiment_pred": "negative",
  "confidence_proxy": 0.81,
  "difficulty_gold": 2,
  "source": "IMDb"
}
```

这里的 `confidence_proxy` 不要求像生成式模型那样是可解释概率，它只是由决策函数分数经后处理得到的辅助展示值，并不参与任何训练目标。

## 5. 编码器主模型：BERT-base 微调 Pipeline (Encoder Fine-tuning Pipeline)

### 5.1 输入表示与 tokenizer 策略

`BERT-base` 阶段直接把每条评论视为一个标准文本分类样本。输入文本默认使用 `text_clean`，通过 Hugging Face tokenizer 编码为 `input_ids`、`attention_mask` 等张量。最大序列长度不应拍脑袋决定，而应参考数据工程阶段统计出的长度分布。若大多数样本落在 256 token 以内，则优先选择 `max_length = 256`；若长评论比例较高，再考虑提升到 `384`。

为了减少 padding 浪费，训练阶段推荐采用动态 padding；为了便于复现，应固定 tokenizer 版本和 truncation 策略，并记录在实验配置文件中。

### 5.2 微调目标与损失函数方法

`BERT-base` 的主训练目标是标准的情感二分类，因此最自然的做法是在 `[CLS]` 表示上接一个线性分类头，并使用 `CrossEntropyLoss` 进行训练。由于本项目后续要做 Qwen 的 DPO 增强，因此 BERT 阶段应保持尽量干净和标准，避免在这里同时引入过多复杂技巧，以免后续无法明确收益来自模型结构还是损失函数工程。

若时间允许，可补充一个高纯度训练实验，即在去除 `ambiguous_flag = true` 的训练子集上再跑一版 BERT，观察标签噪声减少是否会换来更稳定的验证表现。但这一实验应作为补充，而不是主线。

### 5.3 训练配置与 checkpoint 选择方法

较稳妥的 BERT 初始训练配置可以是：

- `learning_rate = 2e-5`
- `batch_size = 16` 或 `32`
- `num_epochs = 3~5`
- `weight_decay = 0.01`
- `warmup_ratio = 0.1`

模型选择标准建议与 SVM 保持一致，即优先根据 `val_balanced` 上的 `Macro-F1` 选择最佳 checkpoint，而不是仅看最后一个 epoch。若训练过程中 `val_macro_f1` 连续若干轮不再提升，应尽早停止并保存最优模型。

### 5.4 BERT 输出资产与误差分析方法

BERT 不仅要输出最佳权重，还应为验证集和测试集导出逐样本预测文件。例如，每条记录保存：

- `id`
- `gold_sentiment`
- `pred_sentiment`
- `pred_prob`
- `difficulty`
- `source`
- `movie_title`

这样后续做混淆矩阵、难度分层统计和错误案例聚类时，就不需要再次回头重跑推理。

## 6. 解码器主模型：小型 Qwen 的 SFT Pipeline (Decoder Fine-tuning Pipeline)

### 6.1 模型选择与工程约束

本项目的小型 Qwen 路线建议优先选择 `Qwen3-0.5B-Instruct` 作为默认实现版本。这样做的原因是：它在显存占用、训练速度和 instruction-following 能力之间比较平衡，适合作为课程项目中的主力生成式模型。若后续算力允许，可在不改变 pipeline 结构的前提下切换到 `Qwen3-1.5B-Instruct`，作为更大规模版本的补充实验。

为了降低训练门槛，Qwen 路线推荐采用参数高效微调方法，例如 `LoRA` 或 `QLoRA`，而不是直接全参数微调。这样既能减少显存压力，也能使 SFT 与 DPO 的切换更加顺滑。

### 6.2 SFT 训练样本构造方法

Qwen 的监督微调不应只学习输出一个单词标签，而应学习生成结构化 JSON。这样做的好处是，后续 DPO 可以直接沿用同一输出空间，CLI 和 demo 也能展示模型的难度判断与简短解释。

推荐的 SFT 输入格式为：

- `instruction`：固定任务说明，要求返回合法 JSON。
- `input`：评论文本。
- `output`：最终审定后的结构化答案。

对于 `output`，建议至少包含：

- `sentiment`
- `difficulty`
- `ambiguous_flag`
- `reasoning`

这里特别约束：即使上游数据中保留了 `confidence` 字段，下游 SFT 目标中也**不**应要求模型生成它。这样可以避免模型把训练容量浪费在模仿一个不稳定的数值字段上。

其中，`reasoning` 的来源可以按以下规则构造：

- 若样本来自 `supreme_judge`，优先使用 `judge_analysis`。
- 若样本来自 `strong_consensus` 或 `weak_consensus_upgraded`，可保留一致侧 Annotator 的简短 reasoning，或用模板化规则生成简洁解释。

### 6.3 SFT Prompt 设计方法

为了保证训练与推理的一致性，Qwen 的 prompt 模板应在 SFT 阶段就固定下来。例如，可采用如下风格：

```text
You are given a movie review. Return only valid JSON with the fields
`sentiment`, `difficulty`, `ambiguous_flag`, and `reasoning`.
`sentiment` must be `positive` or `negative`.
`difficulty` must be 1, 2, or 3.

Review:
{review_text}
```

这个模板的关键点在于：它同时约束输出 schema 和任务边界，避免模型把解释写成散文，或者在推理时突然切换到自由文本风格。只要 SFT 与 DPO 均复用同一模板，后续模型对比就会更加干净。

### 6.4 Qwen-SFT 训练与验证方法

Qwen-SFT 应直接基于 `annotated_reviews.jsonl` 或其训练子集构造 instruction-response pairs，然后在固定随机种子下训练。训练时除常规 loss 外，还应额外监控两个验证指标：

- JSON 解析成功率
- 解析后情感标签的 `Macro-F1`

这样做的原因是：生成式模型可能表面 loss 很低，但实际输出经常格式错乱；若不把“能否稳定输出合法 JSON”纳入验证标准，那么模型在 demo 中很可能表现不稳定。

### 6.5 Qwen-SFT 样本 Demo

下面给出一个适合 Qwen-SFT 的监督样本示例：

```json
{
  "instruction": "You are given a movie review. Return only valid JSON with the fields `sentiment`, `difficulty`, `ambiguous_flag`, and `reasoning`. `sentiment` must be `positive` or `negative`. `difficulty` must be 1, 2, or 3.",
  "input": "Visually overwhelming in the best way, but I still cannot tell whether I loved the movie or just admired the production design.",
  "output": "{\"sentiment\":\"negative\",\"difficulty\":2,\"ambiguous_flag\":false,\"reasoning\":\"The text contains surface praise, but the speaker ultimately denies genuine enjoyment. The dominant sentiment is negative, and the difficulty is medium because the contrast is explicit rather than sarcastic.\"}"
}
```

这个样本同时体现了主任务标签、难度标签与解释字段的统一输出格式，因此能够自然过渡到后续 DPO 阶段。

## 7. 基于分歧样本的 Qwen-DPO 增强方法 (Qwen DPO Enhancement Pipeline)

### 7.1 DPO 数据来源与构造原则

Qwen-DPO 不直接使用普通监督样本，而是使用数据工程阶段已经从分歧样本构造好的 `pairwise_dpo.jsonl`。这类记录来自双模型意见不一致的案例，尤其是 `Zero Consensus` 样本，因此天然包含“哪个回答更接近最终裁决”的偏好监督信号。

每条 DPO 训练记录都包含：

- `prompt`
- `chosen`
- `rejected`

其中，`chosen` 是与最终裁决更一致的结构化 JSON 输出，`rejected` 是较差的候选输出。由于偏好方向来自真实裁决而不是人工主观改写，这种数据对课程项目而言具有较高的性价比。

### 7.2 DPO 初始化与训练顺序方法

DPO 不应直接从原始 Qwen 基座模型开始训练，而应**先完成 Qwen-SFT，再在其基础上进行 DPO**。原因在于，若模型尚未学会基本任务格式与标签定义，DPO 很容易把训练过程变成“格式混乱的偏好拟合”，而不是“在已有能力基础上的偏好校正”。

因此，小型 Qwen 的合理训练顺序为：

1. 使用 `annotated_reviews.jsonl` 做 Qwen-SFT。
2. 选取最佳 SFT checkpoint 或 adapter。
3. 使用 `pairwise_dpo.jsonl` 在该 checkpoint 上继续做 DPO。
4. 与 SFT 模型做同一套验证与测试对比。

### 7.3 DPO 训练目标与稳定性控制

DPO 阶段的核心目标不是再去学习一遍标签空间，而是让模型在面对冲突样本时更倾向于输出“更接近最终裁决的结构化答案”。因此，DPO 的学习率应明显低于 SFT，训练轮数也不宜过大，以免模型过度偏向少量高冲突样本。

较稳妥的工程策略包括：

- 使用较小学习率，如 `5e-7 ~ 1e-6`
- 保持与 SFT 相同的 prompt 模板
- 使用固定长度上限，避免 `chosen/rejected` 长度差异过大
- 在验证时同时检查 JSON 解析率与情感 Macro-F1，防止 DPO 只提升偏好 loss 却破坏主任务性能

### 7.4 DPO 消融实验设计方法

为了证明 DPO 的有效性，至少应完成以下一组纵向消融：

- `Qwen-SFT`
- `Qwen-SFT + DPO`

若时间允许，还可进一步比较：

- 仅使用 `sentiment_conflict` 偏好对
- 使用 `sentiment_conflict + difficulty_conflict` 偏好对

这样可以更清楚地回答两个问题：第一，DPO 是否真的提升了模型；第二，加入难度分歧样本究竟是在提供额外信号，还是在引入额外噪声。

## 8. 评测协议、误差分析与数学验证 (Evaluation, Error Analysis & Validation)

### 8.1 主评测指标设计方法

所有模型都应在同一套协议下评测。主指标建议包括：

- `Accuracy`
- `Precision`
- `Recall`
- `F1-score`
- `Macro-F1`
- `Confusion Matrix`

对于 Qwen 这类生成式模型，还应额外报告：

- JSON 解析成功率
- 合法 schema 命中率

因为生成式模型的可用性不仅取决于标签对不对，还取决于它是否能稳定输出系统可消费的结构化结果。

### 8.2 分层评测方法

为验证数据集构造阶段的难度分级是否有效，所有模型都应至少在以下维度上做分层评测：

- `difficulty = 1 / 2 / 3`
- `source`
- `source_type`
- 文本长度区间

理论预期是：对于绝大多数模型，性能应呈现从 `Level 1` 到 `Level 3` 的下降趋势。若某个模型在总体上变强，但在 `Level 3` 上没有显著改进，那么它的提升可能更多来自对显式情感词的拟合，而不是对复杂语境的真正理解。

### 8.3 定性误差分析方法

定性误差分析不应只展示“模型分错了几条样本”，而应围绕共性现象归纳错误类型。建议至少整理以下类别：

- sarcastic expressions
- mixed sentiment
- subtle contextual reversal
- world-knowledge dependence
- long review truncation

对于 BERT 和 Qwen，还可额外比较“错误但解释看似合理”的案例与“标签正确但 JSON 输出格式失败”的案例，从而区分模型是理解失败还是工程输出失败。

### 8.4 显著性与校准补充分析

若项目时间允许，可使用 `McNemar's Test` 比较 BERT、Qwen-SFT 与 Qwen-DPO 在同一测试集上的差异是否具有统计显著性。对于 BERT 和 SVM，还可补充简单的置信度校准分析；对于 Qwen，则可进一步分析 `reasoning` 质量、schema 稳定性与真实正确率之间的关系，判断它的解释输出是否具有参考价值。

## 9. 推理接口、CLI 与 Demo 集成 (Inference Interface, CLI & Demo Integration)

### 9.1 统一推理接口方法

尽管不同模型的内部形式差异较大，本项目仍建议为 SVM、BERT 和 Qwen 定义统一推理接口，例如统一暴露：

- `predict(text)`
- `predict_batch(texts)`
- `format_output(prediction)`

这样做的好处是：命令行工具和 Gradio demo 可以不关心底层到底调用的是 SVM、BERT 还是 Qwen，只需要读取统一结构的返回值。

### 9.2 命令行测试方法

命令行接口至少应支持：

- 输入单条评论文本
- 指定模型名称
- 返回情感标签
- 对分类模型可选返回置信度代理
- 对 Qwen 返回结构化 JSON

这样既符合课程要求中的 command line testing，也方便团队在开发阶段快速抽样验证模型行为。

### 9.3 Gradio Demo 集成方法

Gradio demo 不需要同时挂载所有模型，但建议至少展示最佳模型，并提供可选的“对比模式”，允许用户在同一条评论上比较 SVM、BERT 与 Qwen 的输出差异。对于 Qwen，demo 中可显示：

- `sentiment`
- `difficulty`
- `reasoning`

这样更能体现本项目相较于普通二分类器的扩展价值。

## 10. 可复现性、工程风控与实验产出 (Reproducibility, Risk Control & Deliverables)

### 10.1 可复现性控制方法

建模阶段应冻结以下要素：

- 数据 split 版本
- tokenizer 版本
- 模型名称与 checkpoint 名称
- 训练超参数
- 随机种子
- prompt 模板版本

所有这些信息都应写入配置文件或实验日志头部，而不是只存在于 notebook 输出或终端历史中。

### 10.2 实验日志与 checkpoint 管理方法

每次训练都应保存以下内容：

- 训练配置
- 最优 checkpoint 路径
- 验证集最优指标
- 测试集结果
- 逐样本预测文件

对于 Qwen 的 SFT 和 DPO，还建议额外保存若干固定测试样本的生成输出，以便快速肉眼比较不同版本之间的行为变化。

### 10.3 主要风险与缓解方法

建模阶段的核心风险主要包括：

- SVM 在简单样本上过强，导致后续提升有限
- BERT 对长文本截断敏感
- Qwen-SFT 输出 JSON 不稳定
- DPO 在小数据上过拟合冲突样本

对应的缓解策略分别是：

- 强制报告分层结果，而不是只看总体分数
- 基于长度分布合理设定 `max_length`
- 在验证集中加入 schema 解析率监控
- 将 DPO 学习率、轮数与训练对数量控制在保守范围内

### 10.4 本阶段最终产物

建模阶段最终建议提交和归档的核心资产包括：

- 一组传统基线模型文件
- 一组 BERT 最优 checkpoint
- 一组 Qwen-SFT adapter
- 一组 Qwen-DPO adapter
- 平衡测试集与自然测试集的结果表
- 分层误差分析文件
- 命令行推理脚本
- Gradio demo 脚本

这些产物共同构成项目的“模型层交付物”，与前一阶段的数据集交付物一一对应。

## 11. 模块化实施与协作接口设计 (Implementation Roadmap & Team Interfaces)

若团队希望高效并行推进建模阶段，建议将整体实现拆分为四个相互对接的模块。

### 11.1 模块 M1：传统基线与稀疏特征实验

模块 M1 负责实现 `TF-IDF + Linear SVM` 的训练、调参与预测导出。其输入是固定的监督训练/验证/测试文件，输出是基线模型权重、验证结果、测试结果与逐样本预测文件。该模块的职责是尽快建立一条稳定、可复现且足够强的比较基线。

### 11.2 模块 M2：BERT-base 微调与编码器评测

模块 M2 负责 BERT 的 tokenizer、dataset loader、训练脚本、checkpoint 选择和分层评测。它的输入是与 M1 完全相同的监督数据，输出是最佳 BERT checkpoint 及其完整评测结果。该模块的核心是建立一个可靠的上下文建模参考线。

### 11.3 模块 M3：Qwen-SFT 与 Qwen-DPO

模块 M3 负责构造 instruction tuning 数据、训练 Qwen-SFT、加载 `pairwise_dpo.jsonl` 并继续进行 DPO 微调。其输出包括 SFT adapter、DPO adapter、结构化输出样例和相关评测文件。该模块是本项目“生成式模型 + 偏好优化”路线的核心。

### 11.4 模块 M4：统一评测、CLI 与 Demo

模块 M4 负责聚合不同模型的预测结果，生成统一结果表、误差分析文件、命令行接口和 Gradio 展示页面。它既是实验汇总模块，也是最终用户可见的系统集成模块。

### 11.5 模块协作方法

各模块之间不应通过口头约定传递信息，而应通过稳定 schema 和固定文件路径对接。推荐在项目开始时就明确：

- 统一数据字段名称
- 统一预测输出 schema
- 统一结果表格式
- 统一模型命名规则

这样团队成员即使并行开发，也能够在最后顺利整合为一个完整、可运行、可展示的情感分析系统。
