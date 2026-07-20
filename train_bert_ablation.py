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

from check_length import combine
from config import SEEDS, MODEL_NAME, LEARNING_RATE, EPOCHS, EFFECTIVE_BATCH, EVAL_BATCH, DATA_DIR


MAX_LENGTH = 512


TRAIN_BATCH, GRAD_ACCUM = 8, 2
assert TRAIN_BATCH * GRAD_ACCUM == EFFECTIVE_BATCH

JSONL_PATH = os.path.join(DATA_DIR, "explanations/competing_wisdom_minimax_m3.jsonl")
RESULTS_JSON = os.path.join(DATA_DIR, "results_ablation.json")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def combine_rumour_only(rec):
    """单侧消融:RUMOUR 侧保留真论证,NOT RUMOUR 侧保留标签、清空内容。
    与双侧唯一差异 = 第二个槽里有没有真论证 → 单变量。
    诊断:若单侧≈双侧,模型没在读论证内容(只认位置/存在),
    即 baseline 的风格捷径在增强层复发;若单侧<双侧,内容被利用了。
    """
    return f"{rec['text']} [SEP] RUMOUR: {rec['rumour_argument']} [SEP] NOT RUMOUR:"


def load_augmented(split_csv_path, jsonl_path):
    df = pd.read_csv(split_csv_path)
    exp = pd.read_json(jsonl_path, lines=True)


    exp = exp[["text", "rumour_argument", "nonrumour_argument"]].drop_duplicates(subset="text")

    merged = df.merge(exp, on="text", how="inner")


    assert len(merged) == len(df), f"行数从 {len(df)} 变成 {len(merged)}"

    merged["combined"] = merged.apply(combine_rumour_only, axis=1)
    return merged


def encode_dataset(df):
    ds = Dataset.from_pandas(df[["combined", "label"]])

    def tokenize(batch):

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
                           output_dir=f"./bert_abl_{seed}", seed=seed)

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