import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score


def run_experiment(train_path, test_path, split_name):
    """在指定的 train/test 上训练 TF-IDF + 逻辑回归,并评测。"""
    print("\n" + "=" * 55)
    print(f"划分方式: {split_name}")
    print("=" * 55)

    # 读数据
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"train: {len(train)} 条 (谣言率 {train['label'].mean():.3f})")
    print(f"test:  {len(test)} 条 (谣言率 {test['label'].mean():.3f})")

    # 文本 → TF-IDF(只在训练集 fit!)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])

    # 训练
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, train["label"])

    # 预测 + 评测
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(
        test["label"], y_pred,
        target_names=["Non-rumour (0)", "Rumour (1)"],
        digits=3,
    ))

    # 返回两个关键指标,方便最后对比
    macro_f1 = f1_score(test["label"], y_pred, average="macro")
    rumour_recall = recall_score(test["label"], y_pred, pos_label=1)
    return macro_f1, rumour_recall


# ==================== 跑两种划分 ====================
# 方案一:随机划分(注意:上次用的是 val 评测,这里统一改用 test 评测,才能和事件划分公平对比)
f1_random, recall_random = run_experiment(
    "data/split_random_train.csv",
    "data/split_random_test.csv",
    "随机分层划分 (random)",
)

# 方案二:按事件划分
f1_event, recall_event = run_experiment(
    "data/split_event_train.csv",
    "data/split_event_test.csv",
    "按事件划分 (event-based)",
)

# ==================== 并排对比 ====================
print("\n" + "=" * 55)
print("对比总结")
print("=" * 55)
print(f"{'划分方式':<20} {'macro-F1':>10} {'谣言 recall':>12}")
print(f"{'随机划分':<20} {f1_random:>10.3f} {recall_random:>12.3f}")
print(f"{'按事件划分':<20} {f1_event:>10.3f} {recall_event:>12.3f}")
print(f"{'下降':<20} {f1_random - f1_event:>10.3f} {recall_random - recall_event:>12.3f}")