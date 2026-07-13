"""
BERT 谣言检测:在随机划分和事件划分上分别微调 bert-base-uncased,并对比。
在 Google Colab (GPU) 上运行。数据从 Google Drive 读取。
"""

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, recall_score, precision_score

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
DATA_DIR = "/content/drive/MyDrive/pheme_data"  # Colab 挂载 Drive 后的路径

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


def run_bert(train_df, test_df, split_name, output_dir):
    """在给定 train/test 上微调一个全新的 BERT 并评测。"""
    print(f"\n{'='*55}\n{split_name}\n{'='*55}")

    train_ds = encode_dataset(train_df)
    test_ds = encode_dataset(test_df)

    # 每次都加载全新模型,保证实验独立
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.evaluate()


def main():
    # 读数据
    rand_train = pd.read_csv(f"{DATA_DIR}/split_random_train.csv")
    rand_test = pd.read_csv(f"{DATA_DIR}/split_random_test.csv")
    event_train = pd.read_csv(f"{DATA_DIR}/split_event_train.csv")
    event_test = pd.read_csv(f"{DATA_DIR}/split_event_test.csv")

    # 两种划分分别训练
    res_random = run_bert(rand_train, rand_test, "随机划分", "./bert_random")
    res_event = run_bert(event_train, event_test, "事件划分", "./bert_event")

    # 汇总
    print("\n\n===== BERT 结果汇总 =====")
    for name, res in [("随机划分", res_random), ("事件划分", res_event)]:
        print(f"{name}: macro_f1={res['eval_macro_f1']:.3f}, "
              f"recall={res['eval_rumour_recall']:.3f}, "
              f"precision={res['eval_rumour_precision']:.3f}")


if __name__ == "__main__":
    main()