"""
PHEME 数据准备:从原始 json 到一张干净的带标签表。
流程:加载9事件 → 去重 → 清洗文本 → 保存 pheme_clean.csv
"""

import json
import re
from pathlib import Path
import pandas as pd

# ==================== 配置 ====================
DATA_ROOT = Path("data/6392078/all-rnr-annotated-threads")  # ← 注意核对你的实际路径
RAW_CSV = "data/pheme_raw.csv"
CLEAN_CSV = "data/pheme_clean.csv"


# ==================== 1. 数据加载 ====================
def read_source_texts(folder_dir, label):
    """遍历一个 rumours/non-rumours 文件夹,返回 [(text, label), ...]。"""
    records = []
    for tweet_folder in Path(folder_dir).iterdir():
        if not tweet_folder.is_dir():          # 跳过 .DS_Store 等非文件夹
            continue
        json_file = tweet_folder / "source-tweets" / (tweet_folder.name + ".json")
        with open(json_file, "r", encoding="utf-8") as f:
            tweet = json.load(f)
        records.append((tweet["text"], label))
    return records


def load_all_events(data_root):
    """遍历全部事件,返回一个带 text/label/event 的 DataFrame。"""
    rows = []
    for event_folder in Path(data_root).iterdir():
        if not event_folder.is_dir():
            continue
        event_name = event_folder.name.replace("-all-rnr-threads", "")
        rumours = read_source_texts(event_folder / "rumours", 1)
        non_rumours = read_source_texts(event_folder / "non-rumours", 0)
        for text, label in rumours + non_rumours:
            rows.append((text, label, event_name))
        print(f"  {event_name}: {len(rumours) + len(non_rumours)} 条")
    return pd.DataFrame(rows, columns=["text", "label", "event"])


# ==================== 2. 去重 ====================
def dedupe(df):
    """先删 label 冲突的 text(全部删),再删完全重复(留一份)。"""
    # 找出同一 text 却有多种 label 的(标注冲突)
    conflict_texts = (
        df.groupby("text")["label"].nunique().loc[lambda x: x > 1].index
    )
    df = df[~df["text"].isin(conflict_texts)]
    print(f"  剔除 {len(conflict_texts)} 条冲突 text")

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  删除 {before - len(df)} 条完全重复")
    return df


# ==================== 3. 文本清洗 ====================
def clean_text(text):
    """URL → [URL];换行/多空格 → 单空格;去首尾空格。"""
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_mentions_hashtags(text):
    """@xxx → @USER;#xxx → xxx(去#留词)。"""
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_pipeline(text):
    """完整清洗流水线,从原文一次跑完。"""
    return clean_mentions_hashtags(clean_text(text))


# ==================== main ====================
def main():
    print("1. 加载全部事件...")
    df = load_all_events(DATA_ROOT)
    print(f"   共 {len(df)} 条")
    df.to_csv(RAW_CSV, index=False)
    print(f"   原始表已存 → {RAW_CSV}")

    print("2. 去重...")
    df = dedupe(df)

    print("3. 清洗文本...")
    df["text_raw"] = df["text"]                      # 备份原文
    df["text"] = df["text_raw"].apply(clean_pipeline)  # text 列变为清洗结果

    # 清洗后可能出现空文本,顺手检查一下
    empty = (df["text"].str.len() == 0).sum()
    print(f"   清洗后空文本: {empty} 条")

    df.to_csv(CLEAN_CSV, index=False)
    print(f"4. 干净表已存 → {CLEAN_CSV}  (共 {len(df)} 条)")


if __name__ == "__main__":
    main()