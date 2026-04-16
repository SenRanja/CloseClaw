
- [此段 To 李玮烨或其他队友 用作数据处理](#此段-to-李玮烨或其他队友-用作数据处理)
    - [本lexicon模型不需要训练，是直接进行测评的](#本lexicon模型不需要训练是直接进行测评的)
    - [数据本身问题](#数据本身问题)
    - [模型本身问题](#模型本身问题)
    - [暂时结论](#暂时结论)
- [How to run](#how-to-run)
    - [pip requirements](#pip-requirements)
    - [outputs](#outputs)
        - [source\_0结论](#source_0结论)
        - [source\_1结论](#source_1结论)


# 此段 To 李玮烨或其他队友 用作数据处理

### 本lexicon模型不需要训练，是直接进行测评的

VADER Sentiment，本质是一个词典（lexicon）+规则系统，已经内置了情感分数（-4 ~ +4），不需要学习数据，所以不需要 training，直接 inference + evaluation。

原始数据路径：`./val/source_0/sft_val.json`和`./val/source_1/sft_val.json`

根据output字段的最后的`\\boxed{positive}` 或者 `\\boxed{negative}` 或者`\\boxed{neutral}`为`positive = 1 neutral=0 negative= -1`

### 数据本身问题

类别不平衡（因为lexicon不需要训练，我没看训练+验证集数据，但此处仅看test数据的话是不平衡的）：

如下仅说明test数据（即直接测评的数据）

source_0
```
boxed{positive} 827
boxed{neutral} 40
boxed{negative} 133
```

source_1
```
boxed{positive} 781
boxed{neutral} 21
boxed{negative} 198
```

已输出为两个csv文件，含`id,review,label,prediction`。

输出的metrics:
```
Accuracy,
Precision, 
Recall, 
F1-score, 
Confusion matrix
ROC curve (ROC已直接出图)
AUC score 
```
[IMDB-Spider.py](../IMDB-Spider.py)
### 模型本身问题[IMDB-Spider.py](../IMDB-Spider.py)

VADER 是 binary分类，倾向设计初衷是： 正 vs 负，不是：正 / 中 / 负

neutral ≈ 接近 0，但是很多句子：
稍微偏正 → 就被判 positive
稍微偏负 → 就被判 negative

### 暂时结论

对 positive（正面） 非常强
对 neutral（中性）几乎失效
对 negative（负面） 中等

Positive（表现很好）
precision ≈ 0.77
recall ≈ 0.88
f1 ≈ 0.82

Negative（中等）
f1 ≈ 0.57 / 0.61
能识别，但不稳定。从 confusion matrix 看：[-1] → 被预测成 positive 很多

Neutral（灾难级）
f1 ≈ 0.04 / 0.06
recall ≈ 0.02 ~ 0.04
基本不会识别 neutral


AUC ≈ 0.6635
说明：有一定区分能力但不强

----

如何运行、结果内容，详见如下

# How to run

### pip requirements

    python.exe -m pip install vaderSentiment numpy scikit-learn matplotlib 

### outputs

##### source_0结论

```
Saved results to result_source_0.csv

===== Evaluation =====
Accuracy: 0.7240
Precision (macro): 0.4987
Recall (macro): 0.4834
F1-score (macro): 0.4801

Classification Report:
              precision    recall  f1-score   support

          -1     0.6102    0.5414    0.5737       133
           0     0.1111    0.0250    0.0408        40
           1     0.7748    0.8838    0.8257       327

    accuracy                         0.7240       500
   macro avg     0.4987    0.4834    0.4801       500
weighted avg     0.6779    0.7240    0.6959       500


Confusion Matrix:
[[ 72   5  56]
 [ 11   1  28]
 [ 35   3 289]]

AUC (multi-class): 0.6635
ROC curve saved as roc_curve.png
```

![](img/2026-04-11-18-46-35.png)

##### source_1结论

```
Saved results to result_source_1.csv

===== Evaluation =====
Accuracy: 0.7060
Precision (macro): 0.5245
Recall (macro): 0.4846
F1-score (macro): 0.4890

Classification Report:
              precision    recall  f1-score   support

          -1     0.7630    0.5202    0.6186       198
           0     0.1111    0.0476    0.0667        21
           1     0.6994    0.8861    0.7818       281

    accuracy                         0.7060       500
   macro avg     0.5245    0.4846    0.4890       500
weighted avg     0.6999    0.7060    0.6871       500


Confusion Matrix:
[[103   4  91]
 [  4   1  16]
 [ 28   4 249]]

AUC (multi-class): 0.6649
ROC curve saved as roc_curve.png
```

![](img/2026-04-11-18-49-50.png)


