"""
TF-IDF + Linear SVM — 三分类情感分析
数据来源：sft_train.json（90% train / 10% val）
         test/source_0/sft_test.json
         test/source_1/sft_test.json
直接点运行即可。
"""

import json
import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ── 配置 ────────────────────────────────────────────────────────────────────
TRAIN_PATH   = "sft_train.json"
TEST_SOURCE0 = os.path.join("test", "source_0", "sft_test.json")
TEST_SOURCE1 = os.path.join("test", "source_1", "sft_test.json")
OUTPUT_DIR   = "svm_output"
RANDOM_SEED  = 42
VAL_RATIO    = 0.1

STR2LABEL   = {"positive": 1, "negative": -1, "neutral": 0}
LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}

# 超参数搜索空间
C_VALUES       = [0.1, 0.5, 1.0, 2.0, 5.0]
NGRAM_OPTIONS  = [(1, 2), (1, 3)]
WEIGHT_OPTIONS = [None, "balanced"]
# ───────────────────────────────────────────────────────────────────────────


def extract_label(output_str):
    match = re.search(r'\\boxed\{(\w+)\}', output_str)
    if match:
        return STR2LABEL.get(match.group(1).lower(), None)
    return None


def extract_text(instruction_str):
    parts = instruction_str.split("\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else instruction_str.strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for item in data:
        text  = extract_text(item["instruction"])
        label = extract_label(item["output"])
        if label is not None:
            records.append({"text": text, "label": label})
    return pd.DataFrame(records)


def load_all_data():
    df_full = load_json(TRAIN_PATH)
    print(f"sft_train.json 共 {len(df_full)} 条，标签分布:")
    for label, name in sorted(LABEL_NAMES.items()):
        print(f"  {name}: {(df_full['label'] == label).sum()} 条")

    np.random.seed(RANDOM_SEED)
    idx      = np.random.permutation(len(df_full))
    n_val    = int(len(df_full) * VAL_RATIO)
    df_train = df_full.iloc[idx[n_val:]].reset_index(drop=True)
    df_val   = df_full.iloc[idx[:n_val]].reset_index(drop=True)
    df_test0 = load_json(TEST_SOURCE0)
    df_test1 = load_json(TEST_SOURCE1)

    print(f"\n训练集: {len(df_train)} 条  |  验证集: {len(df_val)} 条")
    print(f"测试集 source_0: {len(df_test0)} 条  |  测试集 source_1: {len(df_test1)} 条\n")
    return df_train, df_val, df_test0, df_test1


def make_pipeline(ngram_range, class_weight, c_val):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            analyzer="word",
        )),
        ("svm", LinearSVC(
            C=c_val,
            class_weight=class_weight,
            max_iter=2000,
            random_state=RANDOM_SEED,
        )),
    ])


def quick_eval(model, X, y):
    preds    = model.predict(X)
    acc      = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
    return round(acc, 4), round(macro_f1, 4)


def full_eval(model, X, y, split_name):
    preds          = model.predict(X)
    present_labels = sorted(set(y) | set(preds))
    target_names   = [LABEL_NAMES[l] for l in present_labels]
    acc       = accuracy_score(y, preds)
    macro_f1  = f1_score(y, preds, average="macro", zero_division=0)
    precision = precision_score(y, preds, average="macro", zero_division=0)
    recall    = recall_score(y, preds, average="macro", zero_division=0)
    cm        = confusion_matrix(y, preds, labels=present_labels)

    print(f"\n{'='*55}")
    print(f"[{split_name}]")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro-F1  : {macro_f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"\n{classification_report(y, preds, labels=present_labels, target_names=target_names, zero_division=0)}")
    print(f"Confusion Matrix (行=真实, 列=预测):")
    print(f"  标签顺序: {target_names}")
    print(cm)
    return {
        "split": split_name, "accuracy": round(acc,4), "macro_f1": round(macro_f1,4),
        "precision": round(precision,4), "recall": round(recall,4),
        "confusion_matrix": cm.tolist(), "label_order": target_names,
    }


def save_csv(model, texts, y_true, split_name, prefix):
    preds = model.predict(texts)
    ids   = [f"{split_name}_{i+1:06d}" for i in range(len(y_true))]
    out_df = pd.DataFrame({
        "id":         ids,
        "review":     texts,
        "label":      [LABEL_NAMES[l] for l in y_true],
        "prediction": [LABEL_NAMES[p] for p in preds],
    })
    path = os.path.join(OUTPUT_DIR, f"{prefix}_{split_name}.csv")
    out_df.to_csv(path, index=False)
    print(f"预测结果已保存: {path}")


def show_top_features(model, n=20):
    vectorizer    = model.named_steps["tfidf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef          = model.named_steps["svm"].coef_
    classes       = model.named_steps["svm"].classes_
    print(f"\n── 各类别 Top {n} 判别特征词 ──")
    for i, cls in enumerate(classes):
        top_words = feature_names[np.argsort(coef[i])[-n:][::-1]]
        print(f"  {LABEL_NAMES.get(cls, cls)}: {', '.join(top_words)}")


# ── 主流程 ──────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_train, df_val, df_test0, df_test1 = load_all_data()

X_train = df_train["text"].tolist()
y_train = df_train["label"].tolist()
X_val   = df_val["text"].tolist()
y_val   = df_val["label"].tolist()
X_test0 = df_test0["text"].tolist()
y_test0 = df_test0["label"].tolist()
X_test1 = df_test1["text"].tolist()
y_test1 = df_test1["label"].tolist()

# ── 训练前：默认配置（C=1.0, bigram, 无权重）────────────────────────────────
print("=" * 55)
print("【训练前】默认配置 SVM（C=1.0, bigram, 无权重）")
model_default = make_pipeline((1, 2), None, 1.0)
model_default.fit(X_train, y_train)
pre_s0_acc, pre_s0_f1 = quick_eval(model_default, X_test0, y_test0)
pre_s1_acc, pre_s1_f1 = quick_eval(model_default, X_test1, y_test1)
print(f"  source_0  Accuracy={pre_s0_acc:.4f}  Macro-F1={pre_s0_f1:.4f}")
print(f"  source_1  Accuracy={pre_s1_acc:.4f}  Macro-F1={pre_s1_f1:.4f}")
save_csv(model_default, X_test0, y_test0, "source_0", "pre_finetune")
save_csv(model_default, X_test1, y_test1, "source_1", "pre_finetune")

# ── 超参数搜索 ───────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("【超参数搜索】")
summary   = []
model_map = {}
step      = 1

for ngram in NGRAM_OPTIONS:
    for cw in WEIGHT_OPTIONS:
        for c_val in C_VALUES:
            name = f"C={c_val} ngram={ngram[1]} {'balanced' if cw else 'none':8s}"
            pipe = make_pipeline(ngram, cw, c_val)
            pipe.fit(X_train, y_train)
            val_acc, val_f1 = quick_eval(pipe, X_val,   y_val)
            s0_acc,  s0_f1  = quick_eval(pipe, X_test0, y_test0)
            s1_acc,  s1_f1  = quick_eval(pipe, X_test1, y_test1)
            print(f"  [{step:02d}] {name}  val F1={val_f1:.4f}  s0 F1={s0_f1:.4f}  s1 F1={s1_f1:.4f}")
            summary.append({"name": name, "val_f1": val_f1, "val_acc": val_acc,
                            "s0_acc": s0_acc, "s0_f1": s0_f1,
                            "s1_acc": s1_acc, "s1_f1": s1_f1})
            model_map[name] = pipe
            step += 1

# ── 汇总对比表 ───────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("【汇总对比表】")
print(f"  {'方案':<36s}  {'val F1':>7s}  {'s0 F1':>7s}  {'s1 F1':>7s}")
print(f"  {'-'*36}  {'-'*7}  {'-'*7}  {'-'*7}")
best_val_f1 = max(r["val_f1"] for r in summary)
for r in summary:
    marker = " ←最佳" if r["val_f1"] == best_val_f1 else ""
    print(f"  {r['name']:<36s}  {r['val_f1']:>7.4f}  {r['s0_f1']:>7.4f}  {r['s1_f1']:>7.4f}{marker}")

# ── 最佳模型完整评测 ─────────────────────────────────────────────────────────
best       = max(summary, key=lambda r: r["val_f1"])
best_model = model_map[best["name"]]
print(f"\n验证集最佳方案：【{best['name']}】，进行完整评测...")

results = []
results.append(full_eval(best_model, X_val,   y_val,   "val"))
results.append(full_eval(best_model, X_test0, y_test0, "source_0"))
results.append(full_eval(best_model, X_test1, y_test1, "source_1"))
show_top_features(best_model, n=20)

# ── 保存训练后预测 CSV ────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("【保存训练后预测 CSV】")
save_csv(best_model, X_test0, y_test0, "source_0", "post_finetune")
save_csv(best_model, X_test1, y_test1, "source_1", "post_finetune")

# ── 训练前后对比摘要 ─────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("【训练前后对比摘要】")
print(f"  {'':25s}  {'Accuracy':>10s}  {'Macro-F1':>10s}")
pre_rows = [
    ("训练前 source_0", pre_s0_acc, pre_s0_f1),
    ("训练前 source_1", pre_s1_acc, pre_s1_f1),
]
for label, acc, f1 in pre_rows:
    print(f"  {label:25s}  {acc:>10.4f}  {f1:>10.4f}")
for r in results:
    if r["split"] in ("source_0", "source_1"):
        label = f"训练后 {r['split']}"
        print(f"  {label:25s}  {r['accuracy']:>10.4f}  {r['macro_f1']:>10.4f}")

# ── 保存模型和指标 ────────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "svm_pipeline.joblib"))
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "best": best["name"], "full_eval": results},
              f, ensure_ascii=False, indent=2)
print(f"\n模型已保存: {OUTPUT_DIR}/svm_pipeline.joblib")
print(f"指标已保存: {OUTPUT_DIR}/metrics.json")
print(f"训练前 CSV: {OUTPUT_DIR}/pre_finetune_source_0.csv")
print(f"训练前 CSV: {OUTPUT_DIR}/pre_finetune_source_1.csv")
print(f"训练后 CSV: {OUTPUT_DIR}/post_finetune_source_0.csv")
print(f"训练后 CSV: {OUTPUT_DIR}/post_finetune_source_1.csv")
