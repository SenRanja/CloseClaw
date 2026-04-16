import json
import re
import csv
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ====== 路径 ======
INPUT_PATH = "./val/source_1/sft_val.json"
OUTPUT_PATH = "result.csv"

# ====== 初始化 ======
analyzer = SentimentIntensityAnalyzer()

# ====== 提取 review ======
def extract_review(instruction):
    parts = instruction.split("\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else ""

# ====== 提取真实标签 ======
def extract_label(output):
    match = re.search(r'\\boxed\{(positive|negative|neutral)\}', output)
    return match.group(1) if match else "neutral"

# ====== label 映射 ======
label_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

# ====== VADER 预测 ======
def vader_predict(text):
    score = analyzer.polarity_scores(text)
    compound = score["compound"]

    if compound >= 0.05:
        return 1
    elif compound <= -0.05:
        return -1
    else:
        return 0

# ====== 读取数据 ======
results = []
y_true = []
y_pred = []
y_score = []  # 用于ROC（prob-like）

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

for idx, item in enumerate(data):
    review = extract_review(item["instruction"])
    true_label_str = extract_label(item["output"])

    true_label = label_map[true_label_str]

    score = analyzer.polarity_scores(review)
    compound = score["compound"]
    pred_label = vader_predict(review)

    y_true.append(true_label)
    y_pred.append(pred_label)
    y_score.append(compound)  # 连续值用于ROC

    results.append({
        "id": idx,
        "review": review,
        "label": true_label,
        "prediction": pred_label
    })

# ====== 保存 CSV ======
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "review", "label", "prediction"])
    writer.writeheader()
    writer.writerows(results)

print(f"Saved results to {OUTPUT_PATH}")

# =========================
# 📊 Evaluation Metrics
# =========================

print("\n===== Evaluation =====")

# ✅ Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")

# ✅ Precision / Recall / F1
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro'
)

print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-score (macro): {f1:.4f}")

# ✅ 分类报告（更详细）
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# =========================
# 📈 ROC + AUC（多分类）
# =========================

# 转 one-hot
classes = [-1, 0, 1]
y_true_bin = label_binarize(y_true, classes=classes)

# ⚠️ 简化：用 compound 映射成3类概率
y_score_3d = np.zeros((len(y_score), 3))

for i, s in enumerate(y_score):
    # 简单分布（你可以写进report解释）
    y_score_3d[i] = [
        max(0, -s),     # negative
        1 - abs(s),     # neutral
        max(0, s)       # positive
    ]

# AUC
auc = roc_auc_score(y_true_bin, y_score_3d, multi_class="ovr")
print(f"\nAUC (multi-class): {auc:.4f}")

# ROC 曲线
plt.figure()

for i, label in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_3d[:, i])
    plt.plot(fpr, tpr, label=f"class {label}")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()

print("ROC curve saved as roc_curve.png")