"""
实验常量的单一事实来源(single source of truth)。

规矩:
  1. train_bert.py 和 train_bert_augmented.py 必须共用本文件的所有超参数。
     任何一个超参数在两个脚本里各写一份 = 迟早漂移 = 你测的不是你以为的东西,
     而且不会报错。
  2. SEEDS 和超参数一旦跑过第一次正式实验就不许再改。
     跑完看结果不满意再回来调 = optional stopping = 作弊。
     想调?那是另一个实验,老实写成一节。
  3. 本文件先于任何实验结果 commit 并 push —— git 时间戳是"事先定死"的唯一证据。
"""


import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.environ["PHEME_DATA"]     # 没设就当场 KeyError。
                                        # 静默回落到 "data" 会让你读到一批不知哪来的文件,
                                        # 而且照跑不误 —— 那正是最贵的一类错误。
                                        # Mac: .env 里写 PHEME_DATA=data
                                        # Colab: PHEME_DATA=/content/... python xxx.py

# ═══ 绝不可改(改了旧结果全部作废) ═══

# 5 个 seed。选哪五个数无所谓(没有"好 seed"),数量和"事先定死"才重要。
# 5 是 NLP 论文最低标准;省下的算力做单侧消融价值更高。
# baseline 和增强版必须用同一组 —— 配对设计:同一个 seed 下两组的分类头初始化
# 和样本顺序逐字节相同,只有 dropout 因序列长度不同而分岔。
SEEDS = [42, 43, 44, 45, 46]

MODEL_NAME = "bert-base-uncased"
LEARNING_RATE = 2e-5          # BERT 原论文推荐区间里最保守的一个
EPOCHS = 3
EFFECTIVE_BATCH = 16          # 真正要锁死的是"有效 batch",不是 per_device_batch
EVAL_BATCH = 32               # 只影响推理速度,不影响结果;锁死纯粹是为了少一个变量

# 事件划分的 test set。split_data.py / error_analysis.py 也在用这两个名字,
# 三处各写一份 = 三份可能不一致的真相。
TEST_EVENTS = ["ottawashooting", "putinmissing"]

# ═══ 允许不同 ═══

# MAX_LENGTH 故意不在这里:baseline=128、增强版=512,它就是被操纵的那个东西。
# 注意它不是自变量,是自变量逼出来的容器尺寸:
# 纯推文 max=50 token,128 的天花板从没生效过 → baseline 设 128 还是 512
# 编码结果逐字节相同。真正的变量只有一个:输入文本 text vs combined。

# 数据目录随机器变(Mac 是 data/,Colab 是 Drive 挂载点),不是实验变量。
# 写错会当场 FileNotFoundError —— 响亮的错误不需要防,静默的才需要。
DATA_DIR = os.environ["PHEME_DATA"]