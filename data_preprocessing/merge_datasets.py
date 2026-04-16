"""
三数据集整合脚本
整合来源：
  1. Rotten Tomatoes (HuggingFace: cornell-movie-review-data/rotten_tomatoes)
  2. Large Movie Review Dataset (Stanford ACLIMDB)
  3. 自爬 IMDB SQLite 数据库 (imdb.db)

输出：merged_reviews.csv / merged_reviews.parquet（可选）

依赖安装：
  pip install datasets pandas pyarrow
"""

import sqlite3
import os
import re
import pandas as pd

# ── 配置区 ────────────────────────────────────────────────────────────────────
IMDB_DB_PATH   = "imdb_reviews.db"          # 修改为你的 imdb.db 路径
ACLIMDB_PATH   = "aclImdb"          # Stanford 解压后的根目录路径
OUTPUT_CSV     = "merged_reviews.csv"
OUTPUT_PARQUET = "merged_reviews.parquet"  # 设为 None 则不输出 parquet
DEDUP          = True               # 是否按 review_text 去重
RATING_THRESHOLD = 6                # 自爬 IMDB: >= 此分数视为 positive

# ── 1. 加载 Rotten Tomatoes ───────────────────────────────────────────────────
def load_rotten_tomatoes():
    print(">>> [1/3] 加载 Rotten Tomatoes ...")
    from datasets import load_dataset
    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")

    rows = []
    for split_name, split_data in ds.items():
        for item in split_data:
            rows.append({
                "text":   item["text"].strip(),
                "label":  int(item["label"]),   # 0=neg, 1=pos
                "source": "rotten_tomatoes",
                "split":  split_name,           # train / validation / test
                "rating": None,
            })
    df = pd.DataFrame(rows)
    print(f"    Rotten Tomatoes: {len(df)} 条")
    return df


# ── 2. 加载 Stanford ACLIMDB ─────────────────────────────────────────────────
def load_aclimdb(base_path):
    """
    ACLIMDB 目录结构：
      aclImdb/
        train/pos/*.txt   train/neg/*.txt
        test/pos/*.txt    test/neg/*.txt
    文件名形如 123_8.txt，其中 8 是 1-10 分的评分。
    """
    print(">>> [2/3] 加载 Stanford ACLIMDB ...")
    rows = []
    for split in ("train", "test"):
        for sentiment in ("pos", "neg"):
            folder = os.path.join(base_path, split, sentiment)
            if not os.path.isdir(folder):
                print(f"    ⚠ 目录不存在，跳过: {folder}")
                continue
            label = 1 if sentiment == "pos" else 0
            for fname in os.listdir(folder):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(folder, split, sentiment, fname)
                # 从文件名提取评分，例如 '1234_9.txt' → rating=9
                m = re.match(r"\d+_(\d+)\.txt$", fname)
                rating = int(m.group(1)) if m else None
                with open(os.path.join(folder, fname), encoding="utf-8") as f:
                    text = f.read().strip()
                rows.append({
                    "text":   text,
                    "label":  label,
                    "source": "aclimdb",
                    "split":  split,
                    "rating": rating,
                })
    df = pd.DataFrame(rows)
    print(f"    ACLIMDB: {len(df)} 条")
    return df


# ── 3. 加载自爬 IMDB SQLite ───────────────────────────────────────────────────
def load_imdb_db(db_path, rating_threshold=6):
    """
    表结构：id INTEGER, rating INTEGER (1-10), review_text TEXT
    rating >= threshold → positive (1)，否则 negative (0)
    """
    print(">>> [3/3] 加载自爬 IMDB SQLite ...")
    con = sqlite3.connect(db_path)

    # 自动检测表名（取第一张表）
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
    table_name = tables.iloc[0]["name"]
    print(f"    使用表: {table_name}")

    df_raw = pd.read_sql(f"SELECT id, rating, review_text FROM {table_name}", con)
    con.close()

    df_raw = df_raw.dropna(subset=["review_text"])
    df_raw["text"]   = df_raw["review_text"].str.strip()
    df_raw["label"]  = (df_raw["rating"] >= rating_threshold).astype(int)
    df_raw["source"] = "imdb_custom"
    df_raw["split"]  = "train"   # 爬取数据全归 train，可按需改为 None
    df_raw["rating"] = df_raw["rating"]

    df = df_raw[["text", "label", "source", "split", "rating"]].copy()
    print(f"    自爬 IMDB: {len(df)} 条  "
          f"(正样本阈值 rating >= {rating_threshold})")
    return df


# ── 4. 合并 & 清洗 ────────────────────────────────────────────────────────────
def merge_and_clean(dfs, dedup=True):
    print(">>> 合并中 ...")
    merged = pd.concat(dfs, ignore_index=True)

    # 基础清洗
    merged["text"] = merged["text"].str.strip()
    merged = merged[merged["text"].str.len() > 0]

    # 去重（以 text 为键，保留首次出现）
    if dedup:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["text"])
        print(f"    去重: {before} → {len(merged)} 条（移除 {before - len(merged)} 条重复）")

    # 加自增 id 列（从 1 开始）
    merged.insert(0, "id", range(1, len(merged) + 1))
    
    # 统计
    print("\n── 数据集统计 ──────────────────────────────")
    print(merged.groupby(["source", "label"]).size().rename("count").to_string())
    print(f"\n总计: {len(merged)} 条")
    print("────────────────────────────────────────────\n")

    return merged


# ── 5. 保存 ───────────────────────────────────────────────────────────────────
def save(df, csv_path, parquet_path=None):
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV 已保存: {csv_path}  ({os.path.getsize(csv_path) / 1e6:.1f} MB)")
    if parquet_path:
        df.to_parquet(parquet_path, index=False)
        print(f"✅ Parquet 已保存: {parquet_path}  ({os.path.getsize(parquet_path) / 1e6:.1f} MB)")


# ── 主流程 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dfs = []

    # Rotten Tomatoes（需要网络）
    try:
        dfs.append(load_rotten_tomatoes())
    except Exception as e:
        print(f"⚠ Rotten Tomatoes 加载失败，已跳过: {e}")

    # Stanford ACLIMDB（需要本地文件）
    if os.path.isdir(ACLIMDB_PATH):
        try:
            dfs.append(load_aclimdb(ACLIMDB_PATH))
        except Exception as e:
            print(f"⚠ ACLIMDB 加载失败，已跳过: {e}")
    else:
        print(f"⚠ 未找到 ACLIMDB 目录 '{ACLIMDB_PATH}'，已跳过")

    # 自爬 IMDB SQLite
    if os.path.isfile(IMDB_DB_PATH):
        try:
            dfs.append(load_imdb_db(IMDB_DB_PATH, RATING_THRESHOLD))
        except Exception as e:
            print(f"⚠ imdb.db 加载失败，已跳过: {e}")
    else:
        print(f"⚠ 未找到 '{IMDB_DB_PATH}'，已跳过")

    if not dfs:
        print("❌ 没有任何数据集加载成功，退出。")
        exit(1)

    merged = merge_and_clean(dfs, dedup=DEDUP)
    save(merged, OUTPUT_CSV, OUTPUT_PARQUET)
