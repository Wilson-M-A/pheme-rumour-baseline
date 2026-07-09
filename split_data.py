import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/pheme_clean.csv")
print(f"总数据: {len(df)} 条\n")

# ==================== 方案一:随机分层划分 70/15/15 ====================
# 先切出 test(15%),剩下 85% 再切成 train 和 val
# stratify=df["label"] 保证每一份的谣言比例都和整体一致

# 第一刀:85% (train+val) vs 15% (test)
trainval, test = train_test_split(
    df,
    test_size=0.15,
    stratify=df["label"],   # 按 label 分层
    random_state=42,        # 固定随机种子,保证每次划分结果一样(可复现)
)

# 第二刀:把 85% 再切成 train 和 val
# val 要占总体的 15%,而现在 trainval 是总体的 85%,所以 val 占 trainval 的 15/85 ≈ 0.1765
val_ratio = 0.15 / 0.85
train, val = train_test_split(
    trainval,
    test_size=val_ratio,
    stratify=trainval["label"],
    random_state=42,
)

print("===== 方案一:随机分层划分 =====")
for name, part in [("train", train), ("val", val), ("test", test)]:
    ratio = part["label"].mean()
    print(f"  {name:5s}: {len(part):5d} 条, 谣言率 {ratio:.3f}")

# 存成文件
train.to_csv("data/split_random_train.csv", index=False)
val.to_csv("data/split_random_val.csv", index=False)
test.to_csv("data/split_random_test.csv", index=False)


# ==================== 方案二:按事件划分 ====================
test_events = ["ottawashooting", "putinmissing"]

# 用 isin 判断每条属不属于测试事件
event_test = df[df["event"].isin(test_events)]
event_train = df[~df["event"].isin(test_events)]  # ~ 取反:不在测试事件里的

print("\n===== 方案二:按事件划分 =====")
print(f"  train: {len(event_train):5d} 条, 谣言率 {event_train['label'].mean():.3f}  (7个事件)")
print(f"  test:  {len(event_test):5d} 条, 谣言率 {event_test['label'].mean():.3f}  (ottawashooting+putinmissing)")

event_train.to_csv("data/split_event_train.csv", index=False)
event_test.to_csv("data/split_event_test.csv", index=False)

print("\n所有划分已保存到 data/ 下")