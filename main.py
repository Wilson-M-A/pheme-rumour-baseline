import json  # Python 内置的 json 库,专门用来读写 json 文件
from pathlib import Path
import pandas as pd

def read_source_texts(folder_dir, label):
    """
    遍历一个文件夹,读出每条源推的 text,
    并给每条都配上传入的 label(1=谣言, 0=非谣言)。
    返回一个列表,每个元素是 (text, label)。
    """
    records = [] # 收集（文本，标签）对

    folder = Path(folder_dir) #把路径字符串编程Path 对象

    #.iterdir() 列出folder 下所有条目（每个都是一个“推文ID”文件夹）
    for tweet_floder in folder.iterdir():

        #如果这个条目不是文件夹（比如 .DS_Store 这种文件），跳过他
        if not tweet_floder.is_dir():
            continue

        # 拼出这个推文文件夹里 source-tweets 的路径
        source_dir = tweet_floder / "source-tweets"

        # source-tweets里的 json 文件名 = 推文ID + 。json，
        #而推文ID 正好就是文件夹的名字（tweet_folder.name）
        json_file = source_dir / (tweet_floder.name +".json")

        #打开并解析这个json
        with open(json_file, "r",encoding= "utf-8") as f:
            tweet = json.load(f)

        #把 text 取出来，加入列表
        records.append((tweet["text"],label))

    return records

def read_one_event(event_dir):
    """
    读一个事件文件夹(里面有 rumours 和 non-rumours),
    返回这个事件全部记录:[(text, label), ...]。
    """

    event_path = Path(event_dir)

    rumours = read_source_texts(event_path/"rumours",1)
    non_rumours = read_source_texts(event_path/"non-rumours",0)

    return rumours + non_rumours


# ===== 主程序:遍历全部 9 个事件 =====



if __name__ == "__main__":
    root = Path("data/6392078/all-rnr-annotated-threads")

    all_rows = []  # 收集所有事件的所有记录,每条是 (text, label, event)

    # 遍历 root 下的每个事件文件夹
    for event_folder in root.iterdir():
        if not event_folder.is_dir():
            continue

        # 事件名:文件夹名形如 "charliehebdo-all-rnr-threads",
        # 我们把 "-all-rnr-threads" 去掉,只留 "charliehebdo"
        event_name = event_folder.name.replace("-all-rnr-threads", "")

        # 读这个事件的所有记录
        records = read_one_event(event_folder)

        # 给每条记录补上 event 名,变成 (text, label, event)
        for text, label in records:
            all_rows.append((text, label, event_name))

        print(f"{event_name}: {len(records)} 条")

    # 把收集到的所有记录装进 DataFrame,指定三个列名
    df = pd.DataFrame(all_rows, columns=["text", "label", "event"])

    # ===== 看看这张表 =====
    print("\n===== 数据总览 =====")
    print(f"总行数: {len(df)}")
    print("\n前 5 行:")
    print(df.head())  # .head() 默认看前 5 行

    print("\n各标签数量(0=非谣言, 1=谣言):")
    print(df["label"].value_counts())  # 统计每个标签有多少条

    print("\n各标签数量(0=非谣言, 1=谣言):")
    print(df["label"].value_counts())  # 统计每个标签有多少条

    print("\n各事件数量:")
    print(df["event"].value_counts())

    # 把这张表存成 CSV,方便后续步骤直接读取,不用每次重新解析 json
    # index=False 表示不要把 pandas 那个自动行号也存进去
    df.to_csv("data/pheme_raw.csv", index=False)
    print("\n已保存到 data/pheme_raw.csv")



