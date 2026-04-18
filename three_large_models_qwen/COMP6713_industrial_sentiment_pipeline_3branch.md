# COMP6713 情感分析项目：难度分级数据集构造与分析计划书

## 1. 项目背景与目标 (Introduction & Objectives)

本项目旨在解决电影评论情感分析任务。为了避免模型在简单显式样本上过拟合，并更准确地评估高阶深度学习模型，尤其是微调大语言模型，在复杂语境下的真实推理能力，本项目不仅着眼于数据收集，更致力于构建一个**具备严格难度梯度（Difficulty-Graded）、可复现（Reproducible）且防泄漏（Leakage-aware）**的高质量电影评论情感数据集。

本计划书系统阐述了从**既有原始语料整合**、自动化多模型共识打标（Multi-LLM Consensus Annotation）、探索性数据分析（EDA），到数据切分、防数据泄漏、基线验证与工程化可复现控制的全链路方法论。与一般“先抓数据、再训练模型”的松散流程不同，本项目将数据工程视为核心研究对象：每一步都需要明确输入、处理方法、判定规则、输出格式及质量控制标准。

## 2. 原始语料整合、元数据保留与预处理策略 (Raw Corpus Integration, Metadata Retention & Pre-processing)

由于项目当前阶段已不再进行实时爬取，本 pipeline 将从**固定原始语料池**直接启动。该语料池由两部分组成：一类是从公开电影评论数据集中抽样得到的现成样本，另一类是此前已经抓取完成、但尚未完成情感与难度标注的未打标评论集合。因此，本阶段的重点不再是“如何抓到数据”，而是“如何对现有语料做整合、审计、重采样和清洗”，以最大化后续标注与评测的有效性。

### 2.1 原始语料整合与来源审计方法

本项目的数据输入不再来自在线抓取过程，而是来自已经落盘的原始文件，例如公开数据集的 CSV/JSON 文件，以及此前抓取得到的未打标评论文件。实现上，首先对所有输入源执行**统一字段映射（Schema Harmonization）**：将不同文件中的异构字段统一映射为标准字段，例如 `review_id`、`movie_title`、`source`、`source_type`、`rating`、`review_text`、`timestamp`、`user_id/thread_id` 等。

其中，`source_type` 至少区分为两类：

- `public_dataset`：来自公开发布的数据集样本。
- `unlabeled_crawl`：来自此前已完成抓取、但尚未打标的评论样本。

对于公开数据集，如果其原本带有情感标签、评分标签或官方划分信息，这些字段不会直接删除，而是保存在 `source_label`、`source_split` 或 `source_rating` 中作为**隐藏参考元数据**。这些字段在自动化标注阶段必须被屏蔽，不能暴露给 Annotator 或 Judge，以避免标签泄漏；但它们可以在后续一致性分析中作为辅助参考。

除了字段统一外，本阶段还要完成**来源审计（Source Audit）**，即统计每个输入文件的样本量、缺失率、平均长度、平台比例和是否带有外部标签。该审计结果将写入单独的 `corpus_manifest` 文件，用于后续报告说明“最终数据来自哪些原始库、比例如何、哪些字段可用”。

### 2.2 候选样本重采样与困难样本挖掘方法

由于原始语料已经固定，项目不再通过“再去抓更多中间分评论”来提高困难样本比例，而是改为对现有语料池执行**重采样（Resampling）**和**困难样本挖掘（Hard-example Mining）**。

对于带有评分或外部情感强度信息的样本，可先将评分归一化到统一区间，再优先抽取中间评分区间的评论。例如，可将接近中性或中等评分的样本设置更高采样权重，因为这类文本更可能包含转折、混合态度或隐性极性表达。

对于完全未打标且无评分的爬取语料，则可采用弱监督启发式进行困难样本挖掘，常见方法包括：

- 使用廉价情感工具或简单基线模型计算情感边际分数，优先保留置信度低的样本。
- 检测显式转折词、双重否定、反问句和引号反讽等语言现象，优先保留疑似 Level 2/3 样本。
- 使用多个轻量规则或小模型做初步情感判断，优先保留相互分歧较大的评论作为高价值候选。

通过这种方式，pipeline 可以在不重新抓取数据的前提下，从既有原始池中主动“挖出”更有研究价值的样本，而不是被动接受原始分布。

### 2.3 元数据保留与标准化方法

除评论正文外，本项目将统一保留以下元数据：

- `source`：评论来源平台。
- `source_type`：标记样本来自公开数据集还是既有未打标爬取语料。
- `movie_title`：电影名称，用于分组切分与跨电影泛化评估。
- `review_id`：评论唯一标识，便于去重和追踪。
- `rating`：平台原始评分或归一化评分。
- `timestamp`：评论时间，用于分析时间漂移或做时间切分扩展实验。
- `user_id/thread_id`：若平台可获得，则作为潜在组键，防止同一用户风格或同一线程上下文泄漏到不同数据划分。
- `source_label/source_split`：若来自公开数据集且原始文件自带标签或官方划分，则仅作为隐藏参考字段保存，不参与自动化打标输入。

在标准化阶段，所有元数据会被写入统一 schema。对于缺失字段，不采用任意填充，而是显式记录为空值，以免后续分析时将缺失误解为合法类别。

### 2.4 文本清洗方法

文本清洗分为四步进行。第一步是噪声剔除，包括删除 HTML 标签、URL、明显无语义的脚本片段、异常重复标点和平台渲染残留。第二步是规范化处理，包括统一空白字符、修正常见编码问题、保留原始大小写版本，同时生成一个仅供传统稀疏模型使用的标准化文本版本。第三步是长度过滤，通过统计初始长度分布，剔除少于 20 词的短文本与超过 200 词的超长文本。前者通常缺乏充分语境，后者往往包含多个情感转折点，容易将单样本情感标签复杂化到超出二分类边界。

第四步是去重。去重采用两层机制：

- **精确去重 (Exact Deduplication)**：先对标准化文本进行哈希，删除完全重复记录。
- **近重复检测 (Near-duplicate Detection)**：再利用 MinHash、编辑距离或句向量余弦相似度识别高相似文本。对于跨公开数据集与未打标爬取池之间重复出现的内容、跨平台转载、模板化影评和只改动少数词汇的内容，将根据相似度阈值进行聚类，每簇仅保留一条代表样本。

若项目只处理英文评论，还应在清洗阶段加入语言识别，将非英语文本单独剥离，以免影响后续基于英文词汇与句法现象定义的难度规则。

此外，若公开数据集中本身已经带有情感标签，应在本阶段显式执行**标签隔离（Label Masking）**：即在送入自动化标注模型前，从工作副本中移除或隐藏这些标签字段，确保后续 LLM 判断只基于评论文本本身，而不是受外部标签污染。

### 2.5 本阶段输出

本阶段的主要输出包括两部分：第一，是一个结构统一、已清洗、已去重的候选池，例如 `clean_reviews.jsonl` 或 `clean_reviews.parquet`；第二，是一个记录来源、字段、样本量和缺失情况的 `corpus_manifest.json`。前者用于后续自动化标注，后者用于审计、复现和报告撰写。

### 2.6 两类原始来源数据 Demo

为了让后续实现与分工更加明确，下面给出两个来源样本在完成字段统一和清洗后可能呈现的示例。需要注意的是，这两个示例都属于**进入自动化标注之前的候选池数据**，因此尚不包含本项目最终生成的 `sentiment` 和 `difficulty` 字段。

**示例 A：来自公开数据集的样本 (`source_type = public_dataset`)**

```json
{
  "id": "sample_0001",
  "review_id": "imdb_ds_18452",
  "source": "IMDb",
  "source_type": "public_dataset",
  "movie_title": "The Last Duel",
  "review_text": "I wanted to admire the film more than I actually did. The acting is strong, but the pacing keeps the emotion at a distance.",
  "text_clean": "I wanted to admire the film more than I actually did. The acting is strong, but the pacing keeps the emotion at a distance.",
  "rating": 6.0,
  "timestamp": "2023-08-14",
  "user_id": null,
  "thread_id": null,
  "language": "en",
  "source_label": "positive",
  "source_split": "train",
  "source_rating": 6.0
}
```

这个样本的关键特点是：它来自公开数据集，因此可能带有原始标签和原始划分信息；但这些字段仅作为隐藏参考元数据保留，不能输入到后续 Annotator 或 Judge 中。

**示例 B：来自既有未打标语料的样本 (`source_type = unlabeled_crawl`)**

```json
{
  "id": "sample_0002",
  "review_id": "reddit_raw_00981",
  "source": "Reddit",
  "source_type": "unlabeled_crawl",
  "movie_title": "Dune: Part Two",
  "review_text": "Visually overwhelming in the best way, but I still cannot tell whether I loved the movie or just admired the production design.",
  "text_clean": "Visually overwhelming in the best way, but I still cannot tell whether I loved the movie or just admired the production design.",
  "rating": null,
  "timestamp": "2024-03-10",
  "user_id": "reddit_user_381",
  "thread_id": "thread_7742",
  "language": "en",
  "source_label": null,
  "source_split": null,
  "source_rating": null
}
```

这个样本的关键特点是：它没有现成标签，因此完全依赖后续多模型共识打标；同时它保留了 `thread_id` 和 `user_id`，便于之后进行 group-aware split 和防泄漏控制。

## 3. 难度分级与辅助标注体系设计 (Difficulty Rubric & Auxiliary Annotation Design)

为指导大语言模型进行自动化标注，本项目设计了三级语言学难度评估体系。该体系的核心目标不是描述文本表面复杂度，而是衡量“基础线性模型（如 TF-IDF + Logistic Regression）正确分类该样本的难度”。

### 3.1 难度定义方法

- **Level 1 (简单 - 显式情感)**：文本包含明确、强烈的情感极性词汇，例如 *masterpiece*、*awful*、*terrible*，句法结构相对简单，且整体情感极性没有明显反转。这类样本通常可以被词袋模型、情感词典或线性分类器较稳定地处理。
- **Level 2 (中等 - 混合语境)**：文本中存在转折连词（如 *but, although, however*）、双重否定或“优缺点并存”的评价结构。其主要难点不在于词汇罕见，而在于模型必须综合多个局部片段后才能判断主导情感。例如，“表演很强，但剧本拖沓到毁掉整部电影”属于典型 Level 2。
- **Level 3 (困难 - 隐式与讽刺)**：文本缺乏显式情感锚点词，或通过反讽、夸张、隐喻、世界知识以及语篇常识来传递真实态度。这类样本往往需要更强的语义建模能力，例如 “This movie cured my insomnia” 表面上无负面词，却表达强烈负面情感。

在实际使用中，难度不是靠“主观感觉”判断，而是通过明确问题引导模型：如果只给 TF-IDF 词特征和线性边界，这条评论是否仍容易被正确分类？若答案明显是否定的，则难度应向更高等级靠拢。

### 3.2 情感标签定义方法

本项目采用二分类情感标签：`positive` 与 `negative`。在自动化标注时，要求模型先判断评论的**主导情感极性**，再判断难度等级，而不是同时模糊地做综合判断。这样做可以避免模型把“难度高”误写成“情感不确定”。

对于主导情感不清晰的样本，不直接将其视为第三类标签，而是通过辅助字段 `ambiguous_flag` 标注为“存在真实歧义”。后续这些样本可以进入人工复核池，或者在构造高纯度黄金集时被剔除。这个设计既保留复杂样本的信息，又不破坏主任务的二分类设置。

### 3.3 Prompt 设计方法

自动化标注 Prompt 的核心不是“让模型给答案”，而是让模型遵循一套稳定、可执行的 rubric。一个有效的 Prompt 至少应包含四部分：

- 任务定义：明确说明需要输出 `sentiment`、`difficulty`、`confidence` 和 `ambiguous_flag`。
- 难度标准：把 Level 1/2/3 的定义写成清晰的判别规则，而非口号式标题。
- 判断顺序：先判定主导情感，再判定为什么该样本对线性基线容易或困难。
- 输出约束：要求严格返回 JSON，不允许额外自然语言前后缀。

为了减少模型漂移，可以在 Prompt 中加入少量 few-shot 示例，每个难度至少给出 1 条代表样本，并强调“难度等级衡量的是基线模型的分类难度，而非评论写作是否优美”。

### 3.4 辅助字段设计方法

为提高后续误差分析与人工复核效率，自动化标注阶段除主标签外，还将保留以下辅助字段：

- `confidence`：模型对自身情感判断的置信度分数，建议归一化到 `0-1` 区间，后续可用于筛选高风险样本。
- `ambiguous_flag`：标记文本是否存在真实语义歧义，例如情感确实高度平衡、缺乏主导倾向，或依赖外部知识仍无法稳定判断。
- `reasoning`：用于记录裁决依据，支持后续定性分析和错误归因。其长度无需很长，但应足以说明模型为何判为正/负以及为何判定该难度。

其中，`confidence` 与 `ambiguous_flag` 不直接改变最终二分类标签空间，但会作为人工抽检、困难样本筛选和报告讨论的重要元数据。需要额外强调的是：`confidence` 只服务于上游标注审计与样本筛选，不应作为后续监督微调或 DPO 偏好优化的训练目标。

## 4. 自动化双盲打标与 3 分支共识裁决机制 (Dual-Blind Annotation & Three-Branch Consensus Routing)

本项目采用“基于委员会投票的伪标签生成（Committee-based Pseudo-labeling）”架构，以在成本、效率与标注一致性之间取得工程上可落地的平衡。

### 4.1 双模型并发标注方法

对每条进入候选池的评论，系统将同时向两个架构不同、偏好不同的高性价比大语言模型发起请求，例如 Qwen-Max 与 GPT-4o-mini。两个模型接收相同的任务说明和相同的 JSON schema，但彼此独立生成结果。这样做的目的不是简单取平均，而是利用不同模型在语言理解偏好上的差异制造“共识信号”。

在工程实现上，系统应对每条记录生成唯一任务 ID，并异步调用 Annotator A 和 Annotator B。送入模型的输入必须只包含评论文本及必要的任务说明，不能包含 `source_label`、`source_split` 等公开数据集自带标签字段。返回结果后，先进行 JSON 结构校验，再进入共识路由模块。若某一模型返回格式非法，则先进行有限次数的重试；若持续失败，则记录为 `annotation_error` 并进入待补处理队列。

### 4.2 输出结构约束方法

每个标注模型应输出如下逻辑字段：

- `sentiment`
- `difficulty`
- `confidence`
- `ambiguous_flag`
- `reasoning`

采用结构化 JSON 而非自由文本的主要目的有三点。第一，便于后续程序化路由与统计；第二，减少模型因生成冗长解释而导致的格式偏差；第三，使得重试、日志记录与人工复核可以直接围绕字段而非自然语言进行。

### 4.3 三分支共识路由方法

系统在拿到 A/B 两个结果后，按照以下规则进行自动路由：

1. **强共识 (Strong Consensus)**  
   当 `sentiment` 和 `difficulty` 均完全一致时，说明两模型对该样本的主导情感及其复杂度判断均稳定。此类样本直接进入最终数据集，并记录 `verified_by: strong_consensus`。这类记录通常构成黄金样本池的主体。

2. **弱共识 (Weak Consensus)**  
   当两个模型在 `sentiment` 上一致，但 `difficulty` 存在分歧时，系统采用**难度就高原则 (Max-Difficulty Principle)**。即最终难度取两者中的较大值，并记录 `verified_by: weak_consensus_upgraded`。采用这一原则的理由是：若至少有一个模型感知到该样本包含潜在迷惑性，那么从“构造困难数据集”的目标出发，更保守地保留较高难度是合理的。

3. **零共识 (Zero Consensus)**  
   当两个模型在 `sentiment` 上发生冲突时，说明该样本很可能涉及复杂语境、讽刺、知识依赖或真正歧义。此时触发 **LLM-as-a-Judge** 机制，由更强的裁判模型读取原文、A/B 的推理记录和候选标签，并输出终审标签、难度以及错误归因 `judge_analysis`。最终结果记录为 `verified_by: supreme_judge`。

### 4.4 盲审裁判方法

为了降低裁判模型对某个模型品牌或风格的偏见，Judge 输入中不应暴露“哪个输出来自哪个模型”，而应只呈现“候选解释 1”和“候选解释 2”。裁判模型的任务不是“选择更像某个大模型的话”，而是比较哪一个解释更符合原文证据。

对于 Judge 的输出，也要执行 JSON schema 验证，并保留其简短分析文本。若出现极少数“Judge 仍然低置信度且样本同时带有 `ambiguous_flag = true`”的记录，可将其升级到人工复核池，而不是强行写入黄金集。

### 4.5 全过程留痕方法

自动化打标不是一次性黑箱调用，而是可审计的数据生成过程。系统应完整保存：

- Annotator A/B 与 Judge 的原始 JSON 输出。
- 使用的 Prompt 版本号。
- 模型名称与参数设置，例如 `temperature`、`max_tokens`。
- 调用时间、重试次数与错误状态。
- 最终裁决结果及 `verified_by` 字段。

这些日志既是工程容错资产，也是写 methodology 和后续错误分析时最关键的证据来源。

### 4.6 基于分歧样本的 DPO 偏好对数据集构造方法

在完成双模型打标与 Judge 仲裁后，本项目还将把**模型意见不一致的样本**进一步转化为可用于偏好优化训练的数据，即 DPO (Direct Preference Optimization) 所需的 **pairwise preference dataset**。这样做的核心动机是：零共识样本天然包含“两个候选答案谁更优”的监督信号，尤其适合后续对小型 Qwen 模型做偏好对齐训练。

在样本选择上，优先保留 **Zero Consensus** 记录，即 `annotator_a.sentiment != annotator_b.sentiment` 的条目。因为这类样本在主导情感上存在真实冲突，Judge 的终审结果可以为偏好方向提供最清晰的依据。若后续 DPO 数据量不足，可再选择性加入 `sentiment` 一致但 `difficulty` 不一致的 **Weak Consensus** 条目，但应将其标记为次级来源，以免“难度分歧”噪声稀释“情感偏好”信号。

对每条候选记录，系统将保留三层信息：原始评论文本、Annotator A/B 的结构化输出、以及最终裁决标签。随后根据最终标签对两个候选答案进行一致性比对，并生成一条偏好对：

1. `prompt`：由任务指令和 `review_text`/`text_clean` 拼接而成，只包含评论正文和输出格式约束，不包含任何外部金标或来源标签。
2. `chosen`：与最终裁决结果更一致的候选输出。
3. `rejected`：与最终裁决结果更不一致的候选输出。

在判定 `chosen` 与 `rejected` 时，本项目采用**以主任务为中心的加权一致性规则**。其中，`sentiment` 与最终裁决是否一致作为最高优先级信号；若两者在情感上都一致或都不一致，则再比较 `difficulty` 是否一致；若仍难区分，则进一步参考 `ambiguous_flag` 是否匹配。换言之，偏好方向的优先级为：

`sentiment match` > `difficulty match` > `ambiguous_flag match`

这样设计的原因是：本项目的主任务是二分类情感分析，因此偏好学习首先应强化“主导情感判断正确”的输出风格，而不是让模型过度追逐难度标签或解释长度等次要属性。对于一致性得分完全相同、或差距过小的记录，不应强行生成偏好对，而应直接丢弃，以避免向 DPO 训练中注入伪偏好噪声。

最终写入 DPO 数据集的每条样本建议至少包含以下字段：

- `id`
- `source_id`
- `pair_type`
- `prompt`
- `chosen`
- `rejected`
- `verified_by`
- `final_label`

其中，`chosen` 与 `rejected` 推荐直接保存为完整 JSON 字符串，但字段应收缩为下游真正需要学习的部分，即 `sentiment`、`difficulty`、`ambiguous_flag` 和 `reasoning`。上游自动标注阶段产生的 `confidence` 可以继续保留在原始日志中用于审计，但不应被放入 DPO 的目标输出。这样做的好处是：后续 Qwen 在 DPO 阶段学到的是“结构化输出的整体偏好模式”，而不是去模仿一个未必可靠的置信度数值。

## 5. 人工校准、分层抽样与一致性评估 (Human Calibration & Inter-annotator Agreement)

在大规模自动化运行之前，项目将先执行小规模人工校准，以验证 Rubric 与 Prompt 设计是否稳定。

### 5.1 分层 Pilot Test 方法

Pilot Test 不能简单随机抽取 100 条样本，否则很可能被简单样本主导，无法有效检验困难样本上的一致性。因此，建议从候选池中按 `source × sentiment × difficulty` 进行分层抽样。若难度标签尚未最终确定，可先用初始自动标签做粗分层，再由人工复核。

较稳妥的做法是抽取 100-150 条样本，并保证每个来源平台、每个情感类别和每个难度层级均有代表。这样得到的人工校准集更能反映 pipeline 在各子空间中的稳定性。

### 5.2 双人独立标注方法

至少由两位人工标注者独立完成情感与难度判断。标注时不展示模型预测结果，只提供 rubric 与样本文本，以避免“被模型答案带偏”。标注完成后，再对冲突条目进行讨论式裁决，形成最终人工金标准。

人工标注时建议要求标注者同时记录简短理由，尤其是难度等级判断理由。这样在后续修订 rubric 时，不仅能看到“哪里不一致”，还能知道“不一致是因为什么定义不清”。

### 5.3 一致性评估方法

人工标注结束后，可分别对情感标签和难度标签计算 Cohen's Kappa 或 Krippendorff's Alpha。通常而言，情感标签一致性应高于难度标签，因为难度本身是更高层的构念。若情感一致性较低，说明样本定义或二分类任务边界本身存在问题；若情感一致但难度一致性低，则说明 rubric 仍需进一步细化。

在课程项目中，可把 `Kappa > 0.80` 视为情感标签的较强一致性，把 `Kappa > 0.70` 视为难度标签的可接受一致性。若未达标，应优先回到 rubric 和 prompt 层面修订，而不是盲目扩大自动化标注规模。

### 5.4 自动化对齐检验方法

在人工金标准构建完成后，将其与自动化管线输出进行逐条对比，计算：

- 情感标签准确率。
- 难度标签一致率。
- 各难度层上的分层表现。
- 不同来源平台上的误差差异。

如果公开数据集自带可信的原始情感标签，还可以将其作为**外部参考标签**进行辅助对比，但不能直接等同于本项目最终金标准，因为其标注定义未必与本项目的主导情感判定规则完全一致。如果发现自动化系统在某一类样本上持续偏差，例如讽刺样本总被误判为正面，或 Reddit 评论总被误判为高难度，那么应在正式大规模打标前修订 Prompt、few-shot 示例或 difficulty rubric。

## 6. 数据切分、防泄漏与评测协议 (Data Splitting, Leakage Control & Evaluation Protocol)

为了保证实验结论的有效性，本项目将“数据切分”视为方法学核心环节，而非训练前的最后一步。

### 6.1 组感知切分方法

训练集、验证集和测试集的划分将按照 `movie_title`、`source`、`source_type` 以及可能的 `thread_id/user_id` 执行组切分，而非简单按评论行随机切分。原因在于，若同一电影的大量评论同时进入训练集与测试集，模型可能只是在记忆该电影的主题词、角色名或特定讨论背景，而不是真正学会情感判断。

在实现上，可以先为每条记录构造一个组键，例如优先使用 `movie_title`，若存在 `thread_id` 或 `user_id`，则可采用多级组键。对于来自公开数据集且原本已有官方 train/test 划分的样本，不建议未经处理直接混入主实验划分；更稳妥的做法是要么只保留其原始文本参与主池并重新做统一切分，要么将其单独保留为外部参考集合。随后按组而非按行做训练/验证/测试划分，并尽可能维持三者在情感分布与来源平台上的近似平衡。

### 6.2 切分顺序控制方法

所有切分操作应在去重之后、降采样之前完成。若先平衡再切分，可能会使训练集与测试集共享同一相似簇中的样本，或者让测试集分布被训练需求反向塑形，破坏评测独立性。

推荐使用固定随机种子生成一次主切分，并将切分清单写入独立清单文件，以保证后续任何实验都可复用同一划分。

### 6.3 双测试协议设计方法

为兼顾研究公平性与真实世界有效性，本项目将同时维护两类测试集：

- **平衡基准测试集 (Balanced Benchmark Set)**：从测试池中按正负情感和 1/2/3 难度做控制采样，使类别分布尽量均衡。这个集合主要用于观察模型在不同难度层上的纯能力差异。
- **自然分布测试集 (Natural Distribution Test Set)**：保持原始平台评论的自然分布，不人为强行拉平。这个集合用于评估模型在更接近真实业务环境下的表现。

最终报告中，两类测试结果都应呈现。前者回答“模型在受控设置下到底有多强”，后者回答“模型在真实场景中是否仍然可靠”。

### 6.4 评测指标设计方法

除总体 Accuracy 外，建议至少报告以下指标：

- Macro-F1，用于避免类别不均衡时准确率掩盖问题。
- 各难度层上的 Accuracy 与 F1，用于验证难度分级是否成立。
- 各来源平台上的性能分解，用于判断模型是否被平台风格绑架。
- 混淆矩阵，用于分析常见误判方向。

若项目时间允许，还可对基线与改进模型进行显著性检验，例如使用 McNemar's Test 比较两模型在同一测试集上的分类差异是否具有统计显著性。

## 7. 探索性数据分析与数学验证 (Exploratory Data Analysis & Validation)

在数据集定型前，必须通过统计学与机器学习方法对其进行严格审查与“排雷”。

### 7.1 描述性统计与宏观特征检查

首先统计正负向情感比例、Level 1/2/3 难度分布、各来源平台样本量以及评论长度分布。该步骤的目的不是“画图好看”，而是检查数据集是否存在明显倾斜。例如，如果 70% 样本都来自 IMDb，或者 80% 的样本都是 Level 1，那么后续模型可能学到来源偏差或简单词汇规则，而非真正的情感理解能力。

长度分布方面，应绘制直方图并计算中位数、均值和 95% 分位数。这样可以为传统特征提取和深度模型的最大序列长度设定提供依据，而不是拍脑袋决定截断长度。

### 7.2 词汇级特征分析方法

在词汇级分析中，可先使用 TF-IDF 提取 Unigram 与 Bigram 特征，再分别计算正类与负类中权重最高的词组。若发现大量平台特有词、电影标题、角色名称或中性领域词在高权重特征中占据主导位置，则说明模型存在学习“伪相关信号”的风险。

为进一步验证词汇偏差，可结合卡方检验（Chi-square）或互信息（Mutual Information）评估哪些词最能区分类别。若区分度最高的词并非真正情感词，而是平台词或专有名词，则需要扩展停用词表或重新采样，以降低这些捷径特征对模型的污染。

### 7.3 流形学习与高维空间可视化方法

为了从几何上观察难度分级是否合理，可先使用句向量模型或预训练文本编码器提取每条评论的高维表示，再通过 UMAP 或 t-SNE 将其降维到二维空间。若样本量较大，可先用 PCA 降到 50 维后再执行 t-SNE，以降低噪声并提升可视化稳定性。

在可视化时，可分别用颜色表示情感极性，用点形状或透明度表示难度等级。理论预期是：Level 1 样本更接近清晰可分的极性团簇，而 Level 3 样本更可能分布在正负边界附近或呈现弥散状态。如果这一趋势存在，则可作为“LLM 难度打标具有几何支持”的定性证据。

### 7.4 难度标签的基线反向印证方法

仅凭 LLM 给出的难度标签仍不足以完全证明其有效性，因此需要使用传统基线进行反向验证。至少训练两类模型，例如 `TF-IDF + Logistic Regression` 与 `TF-IDF + Linear SVM`。若资源允许，可加入一个轻量级 Transformer 模型，形成“线性基线 vs 神经基线”的对照。

验证时，不仅看总体成绩，更要看各难度层的分层成绩。如果多个基线模型均呈现从 Level 1 到 Level 3 的稳定性能下降，例如 `Level 1 > Level 2 > Level 3`，则说明当前难度分级反映的是一般性建模难度，而非某一个特定模型的偶然偏好。

## 8. 数据集定型、可复现性与工程风控 (Dataset Finalization, Reproducibility & Engineering Safeguards)

### 8.1 数据集定型方法

在完成 EDA、人工校准和基线验证后，训练集可进行重平衡处理。具体做法是对过剩类别执行随机降采样，构建正负面 `1:1`、三个难度 `1:1:1` 的平衡训练集，用于模型开发和受控实验比较。与此同时，验证集与测试集不应只保留平衡版本，而应同时保留自然分布版本，以保证最终结论既有研究上的公平性，也有现实外推性。

最终可产出多份标准化数据文件，例如：

- `train_balanced.jsonl`
- `val_balanced.jsonl`
- `test_balanced.jsonl`
- `val_natural.jsonl`
- `test_natural.jsonl`
- `annotation_logs.jsonl`
- `pairwise_dpo.jsonl`

### 8.2 工程容错方法

自动化打标是一个长时间运行、强依赖外部 API 的过程，因此必须具备容错能力。建议在调用层实现以下机制：

- 对每次请求使用 `try-except` 封装。
- 对临时性错误采用指数退避（Exponential Backoff）重试。
- 对每条已完成样本执行 `.jsonl` 追加写入，而不是等整批结束后统一落盘。
- 为任务状态设置 `pending/running/completed/failed` 字段，支持断点续跑。

这样即使中途网络中断或单个样本调用失败，也不会破坏整个批次的处理结果。

### 8.3 可复现性控制方法

为了保证数据工程过程可复现，项目应冻结以下要素：

- Prompt 模板版本。
- JSON schema 版本。
- 模型名称与 API 参数，例如 `temperature`、`max_tokens`。
- 随机种子。
- 数据切分清单。

所有版本信息都应写入 manifest 文件或日志头部。这样当最终结果需要复查时，团队可以明确知道“这份数据集是由哪一版 Prompt、哪一个模型组合、哪一次切分策略产生的”。

### 8.4 定性资产归档方法

零共识样本中的 `judge_analysis` 是最终报告中极有价值的定性材料。建议将其单独导出成专题文件，并按现象进行分组，例如“讽刺表达”“优缺点冲突”“依赖世界知识”“语义真歧义”等。这样在写 final report 的错误分析章节时，就不必再临时回头翻日志。

### 8.5 最终数据集 Demo

完成双模型标注、3 分支共识路由和必要的 Judge 仲裁后，最终写入 `annotated_reviews.jsonl` 或后续训练/评测文件的样本会比候选池数据多出一层“本项目内部生成的标签与审计信息”。下面给出一个最终样本示例：

```json
{
  "id": "sample_0002",
  "review_id": "reddit_raw_00981",
  "source": "Reddit",
  "source_type": "unlabeled_crawl",
  "movie_title": "Dune: Part Two",
  "review_text": "Visually overwhelming in the best way, but I still cannot tell whether I loved the movie or just admired the production design.",
  "text_clean": "Visually overwhelming in the best way, but I still cannot tell whether I loved the movie or just admired the production design.",
  "rating": null,
  "timestamp": "2024-03-10",
  "user_id": "reddit_user_381",
  "thread_id": "thread_7742",
  "language": "en",
  "sentiment": "negative",
  "difficulty": 2,
  "confidence": 0.79,
  "ambiguous_flag": false,
  "verified_by": "supreme_judge",
  "annotator_a": {
    "sentiment": "positive",
    "difficulty": 2,
    "confidence": 0.63,
    "ambiguous_flag": true,
    "reasoning": "The reviewer praises the visuals and sounds impressed overall, even though some hesitation remains."
  },
  "annotator_b": {
    "sentiment": "negative",
    "difficulty": 2,
    "confidence": 0.74,
    "ambiguous_flag": false,
    "reasoning": "The core judgment is that admiration does not translate into genuine enjoyment, so the dominant sentiment is negative."
  },
  "judge_analysis": "The text contains surface praise, but the speaker ultimately denies genuine enjoyment. The dominant sentiment is negative, and the difficulty is medium because the contrast is explicit rather than sarcastic.",
  "annotation_status": "completed"
}
```

这个最终样本体现了三点。第一，原始来源字段仍然被完整保留，因此后续仍可按平台、电影和线程做分析。第二，本项目生成的核心标签包括 `sentiment`、`difficulty`、`confidence`、`ambiguous_flag` 和 `verified_by`。第三，若样本经过零共识分支仲裁，则会额外保留 `annotator_a`、`annotator_b` 与 `judge_analysis`，方便后续做错误归因与定性分析；若样本属于强共识或弱共识分支，则 `judge_analysis` 可为空。

### 8.6 DPO 偏好对数据集 Demo

在将零共识样本进一步转化为 DPO 偏好对后，最终写入 `pairwise_dpo.jsonl` 的记录将不再直接保存“单一最终答案”，而是保存一个**同一输入下的优选输出与劣选输出对**。下面给出一个由上方 `sample_0002` 转换得到的示例：

```json
{
  "id": "sample_0002:sentiment_conflict:annotator_b_over_annotator_a",
  "source_id": "sample_0002",
  "pair_type": "sentiment_conflict",
  "verified_by": "supreme_judge",
  "prompt": "You are given a movie review. Return only valid JSON with the fields `sentiment`, `difficulty`, `ambiguous_flag`, and `reasoning`. `sentiment` must be `positive` or `negative`. `difficulty` must be 1, 2, or 3.\n\nReview:\nVisually overwhelming in the best way, but I still cannot tell whether I loved the movie or just admired the production design.",
  "chosen": "{\"sentiment\":\"negative\",\"difficulty\":2,\"ambiguous_flag\":false,\"reasoning\":\"The core judgment is that admiration does not translate into genuine enjoyment, so the dominant sentiment is negative.\"}",
  "rejected": "{\"sentiment\":\"positive\",\"difficulty\":2,\"ambiguous_flag\":true,\"reasoning\":\"The reviewer praises the visuals and sounds impressed overall, even though some hesitation remains.\"}",
  "final_label": {
    "sentiment": "negative",
    "difficulty": 2,
    "ambiguous_flag": false
  },
  "preference_reason": "annotator_b is more consistent with the final adjudicated label than annotator_a."
}
```

这个 pairwise 样本体现了 DPO 数据构造的三个关键原则。第一，`prompt` 与原始标注阶段保持同一任务边界，因此偏好优化不会改变任务定义，只会改变模型在冲突样本上的输出偏好。第二，`chosen` 与 `rejected` 并不是人工凭主观感觉编写的新答案，而是直接来自双模型打标阶段已经产生的两个候选输出，因此额外标注成本较低。第三，偏好方向由最终裁决标签决定，因此这类数据本质上是“从自动化仲裁日志中再挖掘出的一层训练信号”。同时，示例中刻意去除了 `confidence` 字段，以确保后续小型 Qwen 模型学习的是标签与解释偏好，而不是数值置信度模仿。

## 9. 模块化实施与协作接口设计 (Implementation Roadmap & Team Interfaces)

若项目需要在短时间内并行推进，建议将整体工作拆分为三个模块，并为每个模块定义清晰的输入输出接口。

### 9.1 模块 A：原始语料整合与数据清洗

模块 A 不再负责在线爬取，而是负责读取公开数据集样本与既有未打标爬取语料，完成字段标准化、来源审计、标签隔离、文本清洗、去重以及候选池输出。它的输入是现成的原始文件，输出是统一格式的 `clean_reviews.jsonl/parquet` 与 `corpus_manifest.json`。该模块的关键质量指标包括字段完整率、跨源去重质量、标签隔离是否彻底以及来源分布控制效果。

### 9.2 模块 B：自动化打标流水线

模块 B 读取模块 A 输出的清洗后数据，调用双模型标注器、执行三分支共识路由，并在必要时触发 Judge 仲裁。它的输出包括最终标注结果文件、原始日志文件和高风险人工复核队列。该模块的核心在于 JSON 解析鲁棒性、重试机制和日志留痕完整性。

### 9.3 模块 C：特征分析与验证

模块 C 负责 EDA、词汇级分析、流形可视化、基线建模、难度验证和图表生成。它的输入是模块 B 的最终标注数据，输出包括分析图、实验表格和可直接放入报告的结论摘要。该模块是“证明整个 pipeline 有效”的核心。

### 9.4 模块协作方法

三个模块之间应通过稳定的数据 schema 对接，而不是口头约定字段名。建议在项目早期就定义统一字段表和文件命名规范，例如每条记录必须包含 `id`、`text_clean`、`sentiment`、`difficulty`、`verified_by` 等关键字段。这样即便团队成员并行开发，也能保证最终整合成本较低。

这种模块化拆分能够显著降低协作耦合度，同时让报告中的“数据工程”“实验设计”和“分析验证”三个部分形成清晰分工。
