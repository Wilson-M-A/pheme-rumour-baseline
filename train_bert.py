"""
BERT 谣言检测:在随机划分和事件划分上分别微调 bert-base-uncased,并对比。
在 Google Colab (GPU) 上运行。数据从 Google Drive 读取。
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
    set_seed,
)
from sklearn.metrics import f1_score, recall_score, precision_score

from config import SEEDS, MODEL_NAME, LEARNING_RATE, EPOCHS, EFFECTIVE_BATCH, EVAL_BATCH, DATA_DIR

MAX_LENGTH = 128          # baseline 专属。故意不进 config —— 它是被操纵的那个东西。
                          # 推文 max=50 token,这个天花板从没生效过。

TRAIN_BATCH, GRAD_ACCUM = 16, 1
assert TRAIN_BATCH * GRAD_ACCUM == EFFECTIVE_BATCH   # 有效 batch 必须是 16。
                                                     # 哪天 OOM 顺手改小 batch,这行当场拦住你。

RESULTS_JSON =  os.path.join(DATA_DIR, "results_baseline.json")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode_dataset(df):
    """pandas DataFrame → 编码好的 HuggingFace Dataset。"""
    ds = Dataset.from_pandas(df[["text", "label"]])

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"
        )

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
    """在给定 train/test 上微调一个全新的 BERT 并评测。"""
    print(f"\n{'='*55}\n{split_name} | seed={seed}\n{'='*55}")

    train_ds = encode_dataset(train_df)
    test_ds = encode_dataset(test_df)

    # ↓↓↓ bug 9.1 的修法:必须在 from_pretrained 之前。
    #     from_pretrained(num_labels=2) 会当场随机初始化分类头(768→2),
    #     而 args.seed 要到 Trainer.__init__ 才生效 —— 那时头早就填完了。
    set_seed(seed)

    # 每次都加载全新模型,保证实验独立
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
        seed=seed,              # 管样本顺序和 dropout。
                                # 不写的话 Trainer 会用默认值 42 把你上面 set 的重置掉,
                                # 5 个 seed 的洗牌顺序就全一样了。
        save_strategy="no",     # 只要分数不要 checkpoint;5 份 440MB Colab 存不下
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.evaluate()


def load_results(path):
    """已完成的结果。Colab 会断线,断了不能从头再来。"""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    splits = {
        "随机划分": ("split_random_train.csv", "split_random_test.csv"),
        "事件划分": ("split_event_train.csv", "split_event_test.csv"),
    }

    results = load_results(RESULTS_JSON)

    for split_name, (train_file, test_file) in splits.items():
        train_df = pd.read_csv(f"{DATA_DIR}/{train_file}")
        test_df = pd.read_csv(f"{DATA_DIR}/{test_file}")

        for seed in SEEDS:
            key = f"{split_name}|{seed}"
            if key in results:
                print(f"skip {key} (已完成)")
                continue

            res = run_bert(train_df, test_df, split_name,
                           output_dir=f"./bert_{seed}", seed=seed)

            results[key] = {
                "split": split_name,
                "seed": seed,
                "macro_f1": res["eval_macro_f1"],
                "rumour_recall": res["eval_rumour_recall"],
                "rumour_precision": res["eval_rumour_precision"],
            }
            # 每个 seed 跑完立刻落盘 —— 断在第 8 次也只丢一次
            with open(RESULTS_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # ═══ 汇总 ═══
    print(f"\n\n{'='*55}\nBERT baseline | {len(SEEDS)} seeds\n{'='*55}")
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