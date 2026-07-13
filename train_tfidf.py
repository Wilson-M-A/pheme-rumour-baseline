import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ==================== 1. 读入划分好的数据 ====================
train = pd.read_csv("data/split_random_train.csv")
val = pd.read_csv("data/split_random_val.csv")

print(f"训练集: {len(train)} 条")
print(f"验证集: {len(val)} 条")

# 取出文本(X)和标签(y)
X_train_text = train["text"]
y_train = train["label"]
X_val_text = val["text"]
y_val = val["label"]

# ==================== 2. 文本 → TF-IDF 数字 ====================
# 创建一个 TF-IDF 转换器
vectorizer = TfidfVectorizer(
    max_features=5000,   # 只保留最重要的 5000 个词,控制规模
    ngram_range=(1, 2),  # 同时考虑单个词和相邻两词组合(如 "not true")
)

# 关键:用【训练集】学习词表和 IDF,然后转换训练集
X_train = vectorizer.fit_transform(X_train_text)

# 验证集只做【转换】,不重新学习(必须用训练集学到的词表!)
X_val = vectorizer.transform(X_val_text)

print(f"\nTF-IDF 特征维度: {X_train.shape}")  # (样本数, 特征数)

# ==================== 3. 训练逻辑回归 ====================
model = LogisticRegression(
    max_iter=1000,        # 最大迭代次数,给足让它收敛
    class_weight="balanced",  # ⭐ 关键:自动给少数类(谣言)更高权重,对抗不平衡
)
model.fit(X_train, y_train)  # 训练!就这一行

print("\n模型训练完成")

# ==================== 4. 在验证集上预测 ====================
y_pred = model.predict(X_val)

# 先粗看一下:预测对了多少
accuracy = (y_pred == y_val).mean()
print(f"验证集准确率(accuracy): {accuracy:.3f}")

print("\n" + "=" * 50)
print("详细评测报告")
print("=" * 50)

# ==================== 分类报告:precision / recall / f1 ====================
print("\n分类报告:")
print(classification_report(
    y_val, y_pred,
    target_names=["Non-rumour (0)", "Rumour (1)"],
    digits=3,
))

# ==================== 混淆矩阵 ====================
print("混淆矩阵:")
cm = confusion_matrix(y_val, y_pred)
print(cm)
print("""
读法:
              预测:非谣言   预测:谣言
真实:非谣言      TN           FP
真实:谣言        FN           TP
""")