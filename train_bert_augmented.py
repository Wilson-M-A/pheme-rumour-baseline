"""
BERT + competing wisdom 增强版。

与 train_bert.py 的唯一差异:输入文本 text → combined,以及由此逼出的容器尺寸 512。
其余超参数全部来自 config.py,不得改动。
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import f1_score, recall_score, precision_score

from check_length import combine          # 拼接格式的单一事实来源,不许重敲
from config import SEEDS, MODEL_NAME, LEARNING_RATE, EPOCHS, EFFECTIVE_BATCH, EVAL_BATCH, DATA_DIR

# 【改动 1】128 → 512。实测 combined max=405,512 零截断。
# 这不是自变量,是自变量逼出来的容器尺寸。
MAX_LENGTH = 512

# 【改动 2】8×2,有效 batch 仍是 16 —— 装的是同一批样本,不是新变量。
TRAIN_BATCH, GRAD_ACCUM = 8, 2
assert TRAIN_BATCH * GRAD_ACCUM == EFFECTIVE_BATCH

JSONL_PATH = os.path.join(DATA_DIR, "explanations/competing_wisdom_minimax_m3.jsonl")
RESULTS_JSON = os.path.join(DATA_DIR, "results_augmented.json")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_augmented(split_csv_path, jsonl_path):
    df = pd.read_csv(split_csv_path)
    exp = pd.read_json(jsonl_path, lines=True)

    # drop_duplicates 合法,是因为 label 从未进入生成 prompt(§7 红线):
    # 同一段 text 的两份论证除采样噪声外无差别,留哪份都一样。
    # 若 label 曾进过 prompt,同一段 text 会有两份基于相反 label 的论证 —— 这条路当场死掉。
    exp = exp[["text", "rumour_argument", "nonrumour_argument"]].drop_duplicates(subset="text")

    merged = df.merge(exp, on="text", how="inner")

    # inner + 这行 assert 是一对:inner 单独用会静默丢行,assert 让它变响亮。
    # 就是这行炸出了 5282→5528,查出匿名化引入的 117 条重复。
    assert len(merged) == len(df), f"行数从 {len(df)} 变成 {len(merged)}"

    merged["combined"] = merged.apply(combine, axis=1)
    return merged


def encode_dataset(df):
    ds = Dataset.from_pandas(df[["combined", "label"]])

    def tokenize(batch):
        # 【改动 3】不在这里 padding,交给 collator 按 batch 动态补。
        # 中位 283 补到 512 = 45% 算力烧在 [PAD] 上。
        # 这不改变结果:attention mask 屏蔽 [PAD],logits 逐位相同。纯粹省钱。
        return tokenizer(batch["combined"], truncation=True, max_length=MAX_LENGTH)

    return ds.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "rumour_recall": recall_score(labels, preds, pos_label=1),
        "rumour_precision": precision_score(labels, preds, pos_label=1),
    }


def run_bert(train_df, test_df, split_name, output_dir, seed):
    print(f"\n{'='*55}\n{split_name} | seed={seed}\n{'='*55}")

    train_ds = encode_dataset(train_df)
    test_ds = encode_dataset(test_df)

    # 必须在 from_pretrained 之前 —— 分类头在那一行就被随机初始化了。
    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        logging_steps=50,
        report_to="none",
        seed=seed,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),   # 【改动 3】
    )
    trainer.train()
    return trainer.evaluate()


def load_results(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    # 【改动 4】只跑事件划分 —— 主实验就是它,门槛也只给它定过。
    # 随机划分的用途(量化事件泄漏)在 baseline 阶段已完成。
    splits = {
        "事件划分": ("split_event_train.csv", "split_event_test.csv"),
    }

    results = load_results(RESULTS_JSON)

    for split_name, (train_file, test_file) in splits.items():
        train_df = load_augmented(f"{DATA_DIR}/{train_file}", JSONL_PATH)
        test_df = load_augmented(f"{DATA_DIR}/{test_file}", JSONL_PATH)

        for seed in SEEDS:
            key = f"{split_name}|{seed}"
            if key in results:
                print(f"skip {key} (已完成)")
                continue

            res = run_bert(train_df, test_df, split_name,
                           output_dir=f"./bert_aug_{seed}", seed=seed)

            results[key] = {
                "split": split_name,
                "seed": seed,
                "macro_f1": res["eval_macro_f1"],
                "rumour_recall": res["eval_rumour_recall"],
                "rumour_precision": res["eval_rumour_precision"],
            }
            with open(RESULTS_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*55}\nBERT + competing wisdom | {len(SEEDS)} seeds\n{'='*55}")
    for split_name in splits:
        rows = [v for v in results.values() if v["split"] == split_name]
        if not rows:
            continue
        print(f"\n{split_name}  (n={len(rows)})")
        for metric in ["macro_f1", "rumour_recall", "rumour_precision"]:
            vals = np.array([r[metric] for r in rows])
            print(f"  {metric:18s} {vals.mean():.3f} ± {vals.std(ddof=1):.3f}   "
                  f"{np.round(vals, 3).tolist()}")


if __name__ == "__main__":
    main()