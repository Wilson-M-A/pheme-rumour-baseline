import pandas as pd
import matplotlib.pyplot as plt     # Python 最常用的画图库

# 读入清洗好的数据
df = pd.read_csv("data/pheme_clean.csv")

# ===== 图1:标签分布 =====
# 统计每个标签的数量
label_counts = df["label"].value_counts().sort_index()  # sort_index 让 0 在前 1 在后
print("标签数量:")
print(label_counts)

# 开始画图
plt.figure(figsize=(6, 4))  # 画布大小(宽6高4英寸)

# 画柱状图:x 轴是标签(0/1),y 轴是数量
bars = plt.bar(
    ["Non-rumour (0)", "Rumour (1)"],  # x 轴的两个柱子标签
    label_counts.values,                # 每个柱子的高度
    color=["#4C72B0", "#C44E52"],       # 两个柱子的颜色(蓝、红)
)

# 在每个柱子上方标出具体数字
for bar, count in zip(bars, label_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # 文字的 x 位置(柱子中间)
        bar.get_height(),                    # 文字的 y 位置(柱子顶部)
        str(count),                          # 显示的文字(数量)
        ha="center", va="bottom",            # 水平居中、垂直在上方
    )

plt.title("Label Distribution (PHEME)")  # 标题
plt.ylabel("Count")                       # y 轴名称

plt.tight_layout()  # 自动调整布局,防止文字被裁掉

# 保存图片到 results 文件夹
plt.savefig("results/label_distribution.png", dpi=150)
print("\n图已保存到 results/label_distribution.png")

plt.show()  # 弹窗显示(如果在 PyCharm 里跑,会在窗口显示)


# ===== 图2:文本长度分布 =====
# 算每条清洗后文本的字符长度,存成新列
df["text_len"] = df["text"].str.len()

print("\n文本长度统计:")
print(df["text_len"].describe())  # describe 给出 均值/最小/最大/分位数 等

# 关键检查:有没有长度为 0 的(清洗后变空了)
empty_count = (df["text_len"] == 0).sum()
print(f"清洗后空文本: {empty_count} 条")

plt.figure(figsize=(7, 4))
plt.hist(df["text_len"], bins=40, color="#55A868", edgecolor="white")
plt.title("Tweet Length Distribution (cleaned)")
plt.xlabel("Character length")
plt.ylabel("Number of tweets")
plt.tight_layout()
plt.savefig("results/text_length.png", dpi=150)
print("图已保存 → results/text_length.png")
plt.close()  # 画多张图时,用 close 关掉当前图,避免叠在一起

# ===== 图3:各事件样本数 =====
event_counts = df["event"].value_counts()  # 默认从多到少排序

plt.figure(figsize=(9, 4))
plt.bar(event_counts.index, event_counts.values, color="#8172B3")
plt.title("Sample Count per Event")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")  # 事件名太长,转45度斜着放,不然会挤
plt.tight_layout()
plt.savefig("results/event_counts.png", dpi=150)
print("图已保存 → results/event_counts.png")
plt.close()

# ===== 图4:各事件的谣言比例 =====
# 按事件分组,算每个事件里 label 的平均值(label 是0/1,平均值正好=谣言占比)
rumour_ratio = df.groupby("event")["label"].mean().sort_values(ascending=False)

print("\n各事件谣言比例:")
print(rumour_ratio)

plt.figure(figsize=(9, 4))
plt.bar(rumour_ratio.index, rumour_ratio.values, color="#CCB974")
plt.axhline(df["label"].mean(), color="red", linestyle="--",
            label=f"Overall ({df['label'].mean():.2f})")  # 画一条整体平均线做参照
plt.title("Rumour Ratio per Event")
plt.ylabel("Rumour ratio")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("results/rumour_ratio.png", dpi=150)
print("图已保存 → results/rumour_ratio.png")
plt.close()