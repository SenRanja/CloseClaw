"""
情感数据集清洗脚本
清洗流程：
  Step 1 - 噪声剔除：HTML标签、URL、脚本残留、异常标点、非ASCII字符
  Step 2 - 规范化：空白字符、编码修复、生成 text_normalized 列
  Step 3 - 词级清洗：去除无意义标点、停用词、纯数字、孤立大写缩写
  Step 4 - 长度过滤：保留 20~200 词的文本
  Step 5 - 去重：精确去重（哈希）+ 近重复检测（MinHash）
  Step 6 - 语言识别：剔除非英语文本

依赖安装：
  pip install pandas ftfy langdetect datasketch
"""

import re
import hashlib
import pandas as pd
from ftfy import fix_text
from langdetect import detect, LangDetectException
from datasketch import MinHash, MinHashLSH

# ── 配置区 ────────────────────────────────────────────────────────────────────
INPUT_CSV            = "merged_reviews.csv"
OUTPUT_CSV           = "cleaned_reviews_1.csv"
OUTPUT_DB            = "cleaned_reviews_1.db"
OUTPUT_PARQUET       = "cleaned_reviews.parquet"  # 设为 None 则不输出

MIN_WORDS            = 20       # 最短词数
MAX_WORDS            = 200      # 最长词数
MINHASH_THRESHOLD    = 0.85     # 近重复相似度阈值（0~1，越高越严格）
MINHASH_PERMUTATIONS = 128      # MinHash 置换数，越大越精确但越慢
LANG_DETECT          = True     # 是否启用语言检测（慢，可设 False 跳过）

# 停用词表：高频低意义词（可按需扩充）
STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "one",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "to", "of", "in", "on", "at", "by", "for", "with",
    "and", "or", "but", "so", "yet", "nor",
    "i", "me", "my", "we", "our", "you", "your",
    "it", "its", "they", "them", "their",
    "who", "what", "which", "when", "where", "how",
}

# ── Step 1：噪声剔除 + 非ASCII清除 ───────────────────────────────────────────
def remove_noise(text: str) -> str:
    # HTML 标签
    text = re.sub(r'<[^>]+>', ' ', text)
    # HTML 实体（&amp; &lt; 等）
    text = re.sub(r'&[a-z]+;', ' ', text)
    # URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # 脚本/样式残留
    text = re.sub(r'\{[^}]{0,200}\}', ' ', text)
    # 异常重复标点（3个以上连续相同标点）
    text = re.sub(r'([!?,.\-])\1{2,}', r'\1', text)
    # 非ASCII字符（emoji、宽字节、特殊符号等）全部移除
    text = text.encode('ascii', errors='ignore').decode('ascii')
    # 多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Step 2：规范化 ────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """
    生成标准化版本（用于去重和稀疏模型），保留原始 text 不变：
    - ftfy 修复编码
    - 转小写 + 去标点
    - 统一空白
    """
    text = fix_text(text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Step 3：词级清洗 ──────────────────────────────────────────────────────────
# def clean_tokens(text: str) -> str:
#     """
#     对 text 做 token 级处理，作用于原始 text（保留大小写和句号逗号）：
#     1. 去除无意义独立标点符号（单独出现的 # @ * ~ ^ = | \ / ` 等）
#        —— 保留句号、逗号、感叹号、问号、引号、括号、连字符（它们有语义作用）
#     2. 去除停用词（高频低意义词）
#     3. 去除纯数字 token（如 2024、123）
#     4. 去除孤立全大写缩写（如 FBI、USA、NATO）
#        —— 注意：不删除首字母大写词（人名、地名、片名），保留语义
#     """
#     # 1. 去除无意义独立标点（不在词内、单独成 token 的符号）
#     text = re.sub(r'(?<!\w)[#@*~^=|\\\/`_+<>](?!\w)', ' ', text)

#     tokens = text.split()
#     filtered = []
#     for tok in tokens:
#         # 2. 去除停用词（忽略大小写比较）
#         if tok.lower() in STOPWORDS:
#             continue
#         # 3. 去除纯数字 token（允许带小数点，如 9.5 也去掉）
#         if re.fullmatch(r'\d+(\.\d+)?', tok):
#             continue
#         # 4. 去除孤立全大写缩写（2~6个大写字母，不含小写）
#         #    例：FBI USA NATO → 删除
#         #    例：Spielberg New York → 保留（含小写或首字母大写）
#         if re.fullmatch(r'[A-Z]{2,6}', tok):
#             continue
#         filtered.append(tok)

#     return ' '.join(filtered)


# ── Step 4：长度过滤 ──────────────────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split())


# ── Step 5a：精确去重 ─────────────────────────────────────────────────────────
def exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df["_hash"] = df["text_normalized"].apply(
        lambda t: hashlib.md5(t.encode()).hexdigest()
    )
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
    print(f"    精确去重: {before} → {len(df)} 条（移除 {before - len(df)} 条）")
    return df


# ── Step 5b：近重复检测（MinHash LSH）────────────────────────────────────────
def near_dedup(df: pd.DataFrame,
               threshold: float = MINHASH_THRESHOLD,
               num_perm: int = MINHASH_PERMUTATIONS) -> pd.DataFrame:
    print(f"    近重复检测（阈值={threshold}）...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    keep_mask = [True] * len(df)
    indices = df.index.tolist()

    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 10000 == 0:
            print(f"      处理进度: {i}/{len(df)}")
        m = MinHash(num_perm=num_perm)
        for word in row["text_normalized"].split():
            m.update(word.encode("utf8"))

        key = str(idx)
        result = lsh.query(m)

        if result:
            keep_mask[indices.index(idx)] = False
        else:
            lsh.insert(key, m)

    before = len(df)
    df = df[keep_mask]
    print(f"    近重复去重: {before} → {len(df)} 条（移除 {before - len(df)} 条）")
    return df.reset_index(drop=True)


# ── Step 6：语言检测 ──────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# ── 主流程 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f">>> 读取数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"    原始数据: {len(df)} 条\n")

    # 保存原始文本用于对比
    df["text_original"] = df["text"].astype(str)

    # ── Step 1：噪声剔除 + 非ASCII清除
    print(">>> Step 1: 噪声剔除 + 非ASCII清除 ...")
    df["text"] = df["text"].astype(str).apply(remove_noise)
    df = df[df["text"].str.len() > 0]
    print(f"    处理后: {len(df)} 条")

    # ── Step 2：规范化，生成 text_normalized 列（保留原始 text）
    print("\n>>> Step 2: 规范化 ...")
    df["text_normalized"] = df["text"].apply(normalize)
    print("    text_normalized 列已生成（原始 text 列保留）")

    # # ── Step 3：词级清洗（作用于原始 text）
    # print("\n>>> Step 3: 词级清洗（停用词、无意义标点、纯数字、全大写缩写）...")
    # df["text"] = df["text"].apply(clean_tokens)
    # # 同步更新 text_normalized
    # df["text_normalized"] = df["text"].apply(normalize)
    # df = df[df["text"].str.len() > 0]
    # print(f"    词级清洗后: {len(df)} 条")

    # ── Step 4：长度过滤（基于清洗后词数）
    print(f"\n>>> Step 4: 长度过滤（{MIN_WORDS}~{MAX_WORDS} 词）...")
    df["_wc"] = df["text"].apply(word_count)
    print(f"    词数分布:\n{df['_wc'].describe().to_string()}")
    short = (df["_wc"] < MIN_WORDS).sum()
    long  = (df["_wc"] > MAX_WORDS).sum()
    df = df[(df["_wc"] >= MIN_WORDS) & (df["_wc"] <= MAX_WORDS)]
    df = df.drop(columns=["_wc"])
    print(f"    过滤掉: 过短 {short} 条 + 过长 {long} 条")
    print(f"    长度过滤后: {len(df)} 条")

    # ── Step 5：去重
    print("\n>>> Step 5: 去重 ...")
    df = df.reset_index(drop=True)
    df = exact_dedup(df)
    df = near_dedup(df)

    # ── Step 6：语言检测
    if LANG_DETECT:
        print("\n>>> Step 6: 语言检测（这一步较慢，请耐心等待）...")
        df["lang"] = df["text"].apply(detect_language)
        non_en = (df["lang"] != "en").sum()
        df = df[df["lang"] == "en"].drop(columns=["lang"])
        print(f"    剔除非英语: {non_en} 条")
        print(f"    语言过滤后: {len(df)} 条")
    else:
        print("\n>>> Step 6: 语言检测已跳过（LANG_DETECT=False）")

    # ── 最终统计
    print("\n── 清洗后数据集统计 ────────────────────────────")
    print(df.groupby(["source", "label"]).size().rename("count").to_string())
    print(f"\n总计: {len(df)} 条")
    print("────────────────────────────────────────────────\n")

    # 加自增 id 列          ← 加在这里
    df["id"] = range(1, len(df) + 1)

    # ── 整理列顺序：text_original 紧跟在 text 后面方便对比
    cols = ["id", "text_original", "label", "source", "split", "rating"]
    df = df[[c for c in cols if c in df.columns]]

    # ── 保存 csv
    # df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    # print(f"✅ CSV 已保存: {OUTPUT_CSV}")
    # if OUTPUT_PARQUET:
    #     df.to_parquet(OUTPUT_PARQUET, index=False)
    #     print(f"✅ Parquet 已保存: {OUTPUT_PARQUET}")

    # ── 保存到 SQLite
    import sqlite3
    conn = sqlite3.connect(OUTPUT_DB)
    df.to_sql("reviews", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ SQLite 已保存: {OUTPUT_DB}（表名: reviews，共 {len(df)} 条）")