"""
BERT-base 微调 — 三分类情感分析
数据来源：sft_train.json（90% train / 10% val）
         test/source_0/sft_test.json
         test/source_1/sft_test.json
直接点运行即可。

依赖安装：
    pip install transformers torch scikit-learn pandas
"""

import json
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ── 配置 ────────────────────────────────────────────────────────────────────
TRAIN_PATH      = "sft_train.json"
TEST_SOURCE0    = os.path.join("test", "source_0", "sft_test.json")
TEST_SOURCE1    = os.path.join("test", "source_1", "sft_test.json")
OUTPUT_DIR      = "bert_output"
MODEL_NAME      = "bert-base-uncased"

# 标签映射
STR2LABEL = {"positive": 1, "negative": -1, "neutral": 0}
LABEL2ID  = {-1: 0, 0: 1, 1: 2}
ID2LABEL  = {0: -1, 1: 0, 2: 1}
LABEL_NAMES = {-1: "negative(-1)", 0: "neutral(0)", 1: "positive(1)"}

# 训练超参数
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.1
VAL_RATIO    = 0.1
RANDOM_SEED  = 42
# ───────────────────────────────────────────────────────────────────────────


def extract_label(output_str):
    """从 output 字段里提取 \\boxed{label}"""
    match = re.search(r'\\boxed\{(\w+)\}', output_str)
    if match:
        label_str = match.group(1).lower()
        return STR2LABEL.get(label_str, None)
    return None


def extract_text(instruction_str):
    """从 instruction 字段里提取评论文本"""
    # 格式是 "Classify the sentiment of this movie review:\n\n{text}"
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
    # 加载训练数据
    df_train_full = load_json(TRAIN_PATH)
    print(f"sft_train.json 共 {len(df_train_full)} 条，标签分布:")
    for label, name in sorted(LABEL_NAMES.items()):
        print(f"  {name}: {(df_train_full['label'] == label).sum()} 条")

    # 90% train / 10% val 切分
    np.random.seed(RANDOM_SEED)
    idx     = np.random.permutation(len(df_train_full))
    n_val   = int(len(df_train_full) * VAL_RATIO)
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    df_train = df_train_full.iloc[trn_idx].reset_index(drop=True)
    df_val   = df_train_full.iloc[val_idx].reset_index(drop=True)

    # 加载测试集
    df_test0 = load_json(TEST_SOURCE0)
    df_test1 = load_json(TEST_SOURCE1)

    print(f"\n训练集: {len(df_train)} 条  |  验证集: {len(df_val)} 条")
    print(f"测试集 source_0: {len(df_test0)} 条  |  测试集 source_1: {len(df_test1)} 条\n")
    return df_train, df_val, df_test0, df_test1


# ── Dataset ─────────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(LABEL2ID[self.labels[idx]], dtype=torch.long),
        }


# ── 评测函数 ─────────────────────────────────────────────────────────────────
def quick_eval(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"]
            out   = model(input_ids=ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
    all_preds  = [ID2LABEL[p] for p in all_preds]
    all_labels = [ID2LABEL[l] for l in all_labels]
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return round(acc, 4), round(macro_f1, 4)


def full_eval(model, dataloader, device, split_name, df=None):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"]
            out   = model(input_ids=ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())

    all_preds  = [ID2LABEL[p] for p in all_preds]
    all_labels = [ID2LABEL[l] for l in all_labels]

    present_labels = sorted(set(all_labels) | set(all_preds))
    target_names   = [LABEL_NAMES[l] for l in present_labels]

    acc       = accuracy_score(all_labels, all_preds)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds, labels=present_labels)

    print(f"\n{'='*55}")
    print(f"[{split_name}]")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro-F1  : {macro_f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, labels=present_labels, target_names=target_names, zero_division=0)}")
    print(f"Confusion Matrix (行=真实, 列=预测):")
    print(f"  标签顺序: {target_names}")
    print(cm)

    # 难度分层评测
    if df is not None and "difficulty" in df.columns:
        print(f"\n── 难度分层评测 [{split_name}] ──")
        for level in sorted(df["difficulty"].dropna().unique()):
            idxs = df[df["difficulty"] == level].index.tolist()
            sp = [all_preds[i]  for i in idxs if i < len(all_preds)]
            sl = [all_labels[i] for i in idxs if i < len(all_labels)]
            if len(sp) == 0: continue
            a = accuracy_score(sl, sp)
            f = f1_score(sl, sp, average="macro", zero_division=0)
            print(f"  Level {int(level)} ({len(sp):5d} 条)  Acc={a:.4f}  Macro-F1={f:.4f}")

    # 歧义样本评测
    if df is not None and "ambiguous_flag" in df.columns:
        print(f"\n── 歧义样本评测 [{split_name}] ──")
        for flag in [True, False]:
            mask_df = df["ambiguous_flag"].astype(str).str.upper() == str(flag).upper()
            idxs = df[mask_df].index.tolist()
            sp = [all_preds[i]  for i in idxs if i < len(all_preds)]
            sl = [all_labels[i] for i in idxs if i < len(all_labels)]
            if len(sp) == 0: continue
            a = accuracy_score(sl, sp)
            f = f1_score(sl, sp, average="macro", zero_division=0)
            label = "歧义样本  " if flag else "非歧义样本"
            print(f"  {label} ({len(sp):5d} 条)  Acc={a:.4f}  Macro-F1={f:.4f}")

    # 保存 id / review / label / prediction CSV
    if df is not None:
        source_prefix = split_name if split_name.startswith("source") else split_name
        ids = [f"{source_prefix}_{i+1:06d}" for i in range(len(all_labels))]
        out_df = pd.DataFrame({
            "id":         ids,
            "review":     df["text"].tolist()[:len(all_labels)],
            "label":      [LABEL_NAMES[l].split("(")[0] for l in all_labels],
            "prediction": [LABEL_NAMES[p].split("(")[0] for p in all_preds],
        })
        csv_path = os.path.join(OUTPUT_DIR, f"post_finetune_{split_name}.csv")
        out_df.to_csv(csv_path, index=False)
        print(f"预测结果已保存: {csv_path}")

    return {
        "split": split_name, "accuracy": round(acc,4), "macro_f1": round(macro_f1,4),
        "precision": round(precision,4), "recall": round(recall,4),
        "confusion_matrix": cm.tolist(), "label_order": target_names,
    }


# ── 主流程 ──────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用设备: {device}\n")

# 加载数据
df_train, df_val, df_test0, df_test1 = load_all_data()

# 加载 tokenizer 和模型
print(f"加载 tokenizer 和模型: {MODEL_NAME} ...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

# 构建 DataLoader
train_loader  = DataLoader(ReviewDataset(df_train["text"].tolist(), df_train["label"].tolist(), tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(ReviewDataset(df_val["text"].tolist(),   df_val["label"].tolist(),   tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
test0_loader  = DataLoader(ReviewDataset(df_test0["text"].tolist(), df_test0["label"].tolist(), tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
test1_loader  = DataLoader(ReviewDataset(df_test1["text"].tolist(), df_test1["label"].tolist(), tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)

# 优化器和调度器
optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

def save_predictions_csv(model, dataloader, device, df, split_name, prefix):
    """保存预测结果为 id/review/label/prediction 格式的 CSV"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            ids_t  = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            lbls   = batch["label"]
            out    = model(input_ids=ids_t, attention_mask=mask)
            preds  = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
    all_preds  = [ID2LABEL[p] for p in all_preds]
    all_labels = [ID2LABEL[l] for l in all_labels]
    ids = [f"{split_name}_{i+1:06d}" for i in range(len(all_labels))]
    out_df = pd.DataFrame({
        "id":         ids,
        "review":     df["text"].tolist()[:len(all_labels)],
        "label":      [LABEL_NAMES[l].split("(")[0] for l in all_labels],
        "prediction": [LABEL_NAMES[p].split("(")[0] for p in all_preds],
    })
    csv_path = os.path.join(OUTPUT_DIR, f"{prefix}_{split_name}.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"预测结果已保存: {csv_path}")
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return round(acc, 4), round(macro_f1, 4)


# ── 微调前评测 ───────────────────────────────────────────────────────────────
print("=" * 55)
print("【微调前】评测原始 BERT 并保存预测 CSV")
pre_val_acc,   pre_val_f1   = quick_eval(model, val_loader,   device)
pre_test0_acc, pre_test0_f1 = save_predictions_csv(
    model, test0_loader, device, df_test0.reset_index(drop=True), "source_0", "pre_finetune")
pre_test1_acc, pre_test1_f1 = save_predictions_csv(
    model, test1_loader, device, df_test1.reset_index(drop=True), "source_1", "pre_finetune")
print(f"  val       Accuracy={pre_val_acc:.4f}  Macro-F1={pre_val_f1:.4f}")
print(f"  test_s0   Accuracy={pre_test0_acc:.4f}  Macro-F1={pre_test0_f1:.4f}")
print(f"  test_s1   Accuracy={pre_test1_acc:.4f}  Macro-F1={pre_test1_f1:.4f}")

# ── 训练循环 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"开始微调，共 {EPOCHS} 个 epoch，每 epoch 约 {len(train_loader)} 步...")

best_val_f1 = 0.0
best_epoch  = 0
epoch_log   = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader, 1):
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if step % 50 == 0 or step == len(train_loader):
            print(f"  Epoch {epoch}/{EPOCHS}  Step {step}/{len(train_loader)}  avg_loss={total_loss/step:.4f}")

    val_acc, val_f1 = quick_eval(model, val_loader, device)
    print(f"\n  → Epoch {epoch} 验证集  Accuracy={val_acc:.4f}  Macro-F1={val_f1:.4f}")
    epoch_log.append({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1})

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        model.save_pretrained(os.path.join(OUTPUT_DIR, "best_checkpoint"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_checkpoint"))
        print(f"  ✓ 最佳模型已保存（epoch {epoch}，val Macro-F1={val_f1:.4f}）")

print(f"\n训练完成，最佳 epoch={best_epoch}，val Macro-F1={best_val_f1:.4f}")

# ── 加载最佳模型做完整评测 ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print("加载最佳 checkpoint 进行完整评测...")
best_model = BertForSequenceClassification.from_pretrained(
    os.path.join(OUTPUT_DIR, "best_checkpoint")
)
best_model.to(device)

results = []
results.append(full_eval(best_model, val_loader,   device, "val",      df_val.reset_index(drop=True)))
results.append(full_eval(best_model, test0_loader, device, "source_0", df_test0.reset_index(drop=True)))
results.append(full_eval(best_model, test1_loader, device, "source_1", df_test1.reset_index(drop=True)))

# ── 微调前后对比摘要 ─────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("【微调前后对比摘要】")
print(f"  {'':22s}  {'Accuracy':>10s}  {'Macro-F1':>10s}")
pre = [
    ("微调前 val",      pre_val_acc,   pre_val_f1),
    ("微调前 source_0", pre_test0_acc, pre_test0_f1),
    ("微调前 source_1", pre_test1_acc, pre_test1_f1),
]
for label, acc, f1 in pre:
    print(f"  {label:22s}  {acc:>10.4f}  {f1:>10.4f}")
for r in results:
    label = f"微调后 {r['split']}"
    print(f"  {label:22s}  {r['accuracy']:>10.4f}  {r['macro_f1']:>10.4f}")

# ── 保存指标 ─────────────────────────────────────────────────────────────────
metrics = {
    "pre_finetune": {
        "val":      {"accuracy": pre_val_acc,   "macro_f1": pre_val_f1},
        "source_0": {"accuracy": pre_test0_acc, "macro_f1": pre_test0_f1},
        "source_1": {"accuracy": pre_test1_acc, "macro_f1": pre_test1_f1},
    },
    "epoch_log":  epoch_log,
    "best_epoch": best_epoch,
    "full_eval":  results,
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"\n指标已保存: {OUTPUT_DIR}/metrics.json")
print(f"最佳模型已保存: {OUTPUT_DIR}/best_checkpoint/")
print(f"微调前 CSV: {OUTPUT_DIR}/pre_finetune_source_0.csv")
print(f"微调前 CSV: {OUTPUT_DIR}/pre_finetune_source_1.csv")
print(f"微调后 CSV: {OUTPUT_DIR}/post_finetune_source_0.csv")
print(f"微调后 CSV: {OUTPUT_DIR}/post_finetune_source_1.csv")
