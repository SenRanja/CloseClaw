## 4.2 Baseline Model: TF-IDF + LinearSVC

To establish a simple, interpretable and easily reproducible reference benchmark, this project first adopts TF-IDF + LinearSVC as the baseline model. This baseline aims to measure the performance of traditional sparse text representation methods on the three-class sentiment analysis task and provide a reference for the improvement of the performance of subsequent pre-trained language models.

### Feature extraction: TF-IDF

TF-IDF (Term frequency-Inverse Document Frequency) is a classic text representation method that can construct a high-dimensional sparse vector representation for each comment based on the occurrence Frequency of the term in the current document and its rarity in the entire corpus. In this experiment, the baseline uses the default TF-IDF setting with `ngram_range=(1,2)` to capture unigram and bigram features, with the maximum number of features set to 50,000 and `sublinear_tf=True` enabled to alleviate the dominant effect of high-frequency words.

This setting can capture limited local word-order information through unigram and bigram features. For example, the model can not only recognize individual emotional words, such as "great" or "awful", but also phrases with emotional tendencies, such as "not good" or "well worth". However, TF-IDF still belongs to the sparse representation method based on term statistics and is difficult to model long-distance dependencies, context relationships and complex semantics.

### Classifier: LinearSVC

After obtaining the TF-IDF vector, this project uses LinearSVC for the three-classification task (positive/negative/neutral). Linear support vector machines achieve classification by learning the maximum margin decision boundary in a high-dimensional feature space, and are particularly suitable for high-dimensional sparse data. Therefore, they are often used as strong baseline models in text classification. For multi-classification tasks, LinearSVC adopts the one-vs-rest strategy to learn the decision functions of each category respectively and outputs the final prediction label based on the highest score.

### Baseline setting

Unlike an extensively tuned traditional model, the final baseline in this project uses a fixed default TF-IDF configuration with `ngram_range=(1,2)`. This keeps the baseline simple, interpretable, and easily reproducible, while allowing the comparison with BERT to focus more clearly on the difference between sparse lexical features and contextualized representations.

### Baseline results

The performance of the TF-IDF + LinearSVC baseline on the two test sets is as follows:

| Test Set | Accuracy | Macro-F1 |
| -------- | -------- | -------- |
| source_0 | 0.8600   | 0.6422   |
| source_1 | 0.8860   | 0.6036   |

It can be seen from the results that this baseline model has a strong recognition ability for the two types of explicit emotional comments, positive and negative, but the recognition effect for the neutral category is significantly insufficient. The fundamental reason lies in the fact that neutral comments often contain transitional structures, have both advantages and disadvantages, or have strong semantic ambiguity, and sparse features based on word frequency statistics are difficult to effectively model such complex contexts. This result provides motivation for the subsequent introduction of pre-trained language models.

## 4.3 Pre-trained Transformer Model: BERT

To enhance the model 's ability to model context semantics, negative structures, and complex emotional expressions, this project further employs BERT-base-uncased as the pre-trained Transformer model.

BERT (Bidirectional Encoder Representations from Transformers) is a bidirectional pre-trained language model based on Transformer Encoder. BERT-base-uncased contains 12 layers of Transformer Encoder, 768-dimensional hidden layer representation, and 12 attention heads, with a total parameter size of approximately 110M. Unlike traditional bag-of-words models, BERT utilizes the context information on both the left and right sides simultaneously when encoding each token through a bidirectional self-attention mechanism, thus being able to handle phenomena such as negation, transition, and semantic ambiguity more effectively.

The main reason for choosing BERT for this project is that the emotional expression in film reviews is often not determined by a single emotional word, but relies on more complex context combinations. For example, "not great" and "great" are highly similar in local vocabulary, but their emotional polarities are completely different; Similarly, comments containing "good acting but weak plot" have both positive and negative emotional cues and are overall closer to neutral or mixed sentiment. Compared with TF-IDF + SVM, BERT is more suitable for handling this type of sentiment classification task that relies on context understanding.

In addition, BERT has completed pre-training on large-scale unlabeled corpora and already possesses strong knowledge of general languages. During the fine-tuning stage, only adaptation on the target task data is required to transfer this general semantic representation to the three-classification task of film reviews. Therefore, BERT should theoretically outperform traditional baseline models in the neutral category, which particularly relies on context modeling.

## 4.4 Fine-tuning Strategy

### Data preparation

The data used for fine-tuning is from sft_train.json, which contains a total of 15,418 movie reviews, among which there are 7,145 positive, 7,145 negative, and 1,128 neutral. The data is divided into the training set and the validation set by 90% / 10%, corresponding to:

- Training set: 13,877 items
- Validation set: 1,541 items

The test set consists of two independent sources, namely source_0 and source_1, each with 500 samples.
The label of each sample is parsed from the \boxed{} format in the output field and mapped to:

- positive → 1
- negative → -1
- neutral → 0

### Input representation

The text is first segmented through the BertTokenizer, and then uniformly truncated or padded to a maximum length of 128. Model inputs include:

- input_ids
- attention_mask

In the classification stage, the final hidden state of the [CLS] token is used as the semantic representation of the entire comment, and it is mapped to the logits of three categories through a linear classification header. Subsequently, the category probability is obtained through softmax.

### Loss Function and Optimizer

Using BertForSequenceClassification model training, loss function (CrossEntropyLoss) for cross entropy loss. The optimizer uses AdamW, with the learning rate set to 2e-5 and the weight decay set to 0.01. Both linear preheating and linear attenuation strategies are adopted simultaneously. The warmup ratio is set to 0.1, meaning that within the first 10% of training steps, the learning rate linearly increases from 0 to the target value and then gradually decreases to 0. This strategy helps to reduce the disturbance to the pre-training parameters in the early stage of training and improve the stability of fine-tuning.

### Training Settings

The batch size for model training is 16, with a total of 3 epochs trained, and the gradient clipping threshold is 1.0. At the end of each epoch, Accuracy and Macro-F1 are calculated on the validation set, and the best checkpoint is selected based on the validation set Macro-F1 to avoid relying solely on Accuracy while ignoring the performance of a few classes. The complete training process runs in the Apple M Series chip (MPS) environment, and each epoch takes approximately 15 to 25 minutes.

The baseline is intended to serve as a straightforward reference rather than a heavily optimized traditional model.

### Best checkpoint selection

The performance of the three epochs on the validation set is as follows:

| Epoch | Validation Accuracy | Validation Macro-F1 |
| ----- | ------------------- | ------------------- |
| 1     | 0.888               | 0.692               |
| 2     | 0.898               | 0.713               |
| 3     | 0.885               | 0.742               |

Although the Accuracy of the third epoch is slightly lower than that of the second epoch, its Macro-F1 is the highest. Therefore, the checkpoint of Epoch 3 is ultimately selected as the final model. This indicates that in class imbalance tasks, focusing only on Accuracy may not truly reflect the model's recognition ability for a few classes, while Macro-F1 is more suitable as a model selection metric.

## 4.5 Experimental Results and Comparative Analysis

### 4.5.1 BERT Fine-tuning results

The overall performance of the best BERT model on the two test sets is as follows:

| Test Set | Accuracy | Macro-F1 |
| -------- | -------- | -------- |
| source_0 | 0.894    | 0.755    |
| source_1 | 0.898    | 0.721    |

The results show that after fine-tuning, BERT achieved good classification effects on both independent test sets, indicating that the pre-trained language model can effectively adapt to the three-category classification task of movie review emotions. Compared with the performance in the validation set stage, the model maintained relatively stable Accuracy and Macro-F1 on the test set, indicating that it has a certain generalization ability.

It is worth noting that although the accuracies of both test sets are close to 0.90, this paper pays more attention to the Macro-F1 metric. This is because there is a significant class imbalance in this task, with neutral class samples being far fewer than positive and negative ones. Therefore, using only Accuracy might mask the model's shortcomings in a few classes. In contrast, Macro-F1 assigns the same weight to each category, which can more comprehensively reflect the overall performance of the model in the three-classification task.

### 4.5.2 Comparative Analysis with the baseline model

To further evaluate the performance gains brought by the pre-trained language model, this paper compares the fine-tuned BERT with the TF-IDF + LinearSVC baseline model. The overall results show that the Macro-F1 of BERT on both test sets is higher than that of the baseline model: it increases from 0.642 to 0.755 on source_0 and from 0.604 to 0.721 on source_1. This indicates that, compared with a simple sparse-feature baseline relying on word frequency statistics, BERT has a stronger semantic modeling ability in the three-class sentiment analysis task.

To more clearly explain the sources of the performance differences between the two types of models, this paper further tallies the Precision, Recall and F1 of each category on the merged test set (source_0 + source_1), and the results are as follows:

| Model        | Class    | Precision | Recall | F1   |
| ------------ | -------- | --------- | ------ | ---- |
| TF-IDF + SVM | positive | 0.92      | 0.92   | 0.92 |
| TF-IDF + SVM | negative | 0.81      | 0.93   | 0.86 |
| TF-IDF + SVM | neutral  | 0.40      | 0.05   | 0.09 |
| BERT         | positive | 0.94      | 0.93   | 0.93 |
| BERT         | negative | 0.90      | 0.92   | 0.91 |
| BERT         | neutral  | 0.40      | 0.36   | 0.38 |

It can be seen from the table that both models perform strongly in the positive and negative categories. For the positive class, the F1 values of TF-IDF + SVM and BERT are 0.92 and 0.93 respectively, with a very small gap. For the negative class, BERT's F1 has increased from 0.86 to 0.91. Although there is a certain improvement, it is still not the main source of the difference between the two. This indicates that for comments with relatively clear emotional tendencies and containing significant emotional words, traditional sparse feature models have been able to achieve better classification results, while the advantages of pre-trained language models have not yet been fully demonstrated.

In contrast, the performance difference of the neutral category is the most significant. The recall of TF-IDF + SVM on the neutral class is only 0.05, dropping to 0.00 on source_1, meaning the model almost completely fails to identify neutral samples — nearly all true neutral reviews are misclassified as positive or negative. The F1 score of 0.09 confirms this near-total failure on the neutral category. The precision of BERT on the neutral class is also 0.40, but the recall has increased to 0.36 and the F1 has increased to 0.38, approximately twice that of the baseline model. That is to say, BERT did not sacrifice precision for performance improvement, but significantly enhanced the recall ability for neutral categories while maintaining a similar precision.

This phenomenon indicates that the main source of the overall performance gap between the two types of models does not lie in positive or negative, but in the processing ability for the neutral category. Neutral comments usually contain transitional structures, coexistence of advantages and disadvantages, ambiguous semantics or mixed sentiment expressions. For instance, a comment might contain both positive and negative evaluations, and the overall sentiment does not obviously lean towards any single polarity. For this type of samples, although TF-IDF can capture local phrase information through n-gram, it still lacks the ability to model the overall context and inter-sentence semantic relationships. Therefore, it is easy to misjudge them as positive or negative. In contrast, BERT can better understand the overall semantics and emotional transitions of sentences through bidirectional context representation, thus demonstrating a more obvious advantage in the neutral category.

Overall, the TF-IDF + SVM as the baseline model verified the effectiveness of traditional bag-of-words features in explicit sentiment classification, while the fine-tuned BERT further demonstrated that deep context semantic modeling plays a crucial role in three-category sentiment analysis tasks, especially in the recognition of neutral categories. The improvement of BERT on the overall Macro-F1 essentially stems from its more balanced performance across various categories, and this balance is precisely the core capability emphasized by Macro-F1.