"""
BERT 错误分析(事件划分):
分析漏抓(False Negative)和误报(False Positive)的分布与内容,
诊断模型的系统性偏差。在 Colab 上接着 train_bert.py 的事件划分模型运行。
"""

import numpy as np


def analyze_errors(trainer, test_df, test_ds):
    """对事件划分的 BERT 做双向错误分析。"""
    # 预测
    predictions = trainer.predict(test_ds)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = test_df["label"].values
    test_df = test_df.reset_index(drop=True)

    # ===== 漏抓 FN:真实=谣言(1),预测=非谣言(0) =====
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_tweets = test_df[fn_mask]
    print(f"漏抓的谣言(FN): {fn_mask.sum()} / {(y_true==1).sum()} 条谣言")
    print("FN 事件分布:")
    print(fn_tweets["event"].value_counts())
    print("\nFN 样本(前 15 条,BERT 以为非谣言、其实是谣言):")
    for i, (_, row) in enumerate(fn_tweets.head(15).iterrows()):
        print(f"  {i+1}. [{row['event']}] {row['text']}")

    # ===== 误报 FP:真实=非谣言(0),预测=谣言(1) =====
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_tweets = test_df[fp_mask]
    print(f"\n\n误报(FP): {fp_mask.sum()} / {(y_true==0).sum()} 条非谣言")
    print("FP 事件分布:")
    print(fp_tweets["event"].value_counts())
    print("\nFP 样本(前 15 条,BERT 以为谣言、其实非谣言):")
    for i, (_, row) in enumerate(fp_tweets.head(15).iterrows()):
        print(f"  {i+1}. [{row['event']}] {row['text']}")


# 用法(在 Colab 里,训练完事件划分模型后调用):
# analyze_errors(trainer_event, event_test, event_test_ds)