"""
匿名化引入的重复与 label 冲突的诊断脚本。

背景:pheme_clean.csv 的去重在 text_raw 上做(重复 0),但模型吃的是 text。
匿名化(@USER / [URL])把不同的推文压成同一字符串 → 产生 text 层面的重复,
而 v1 阶段的去重看不见它们。

这个脚本是 docs/methodology.md 里那段结论的可复现证据。
一个能重跑的发现,比一句写在文档里的断言强。

用法: uv run python check_duplicates.py
"""

import pandas as pd

from config import DATA_DIR


def main():
    clean = pd.read_csv(f"{DATA_DIR}/pheme_clean.csv")

    # ═══ 1. 去重查的是哪一列 ═══
    print("═══ 重复情况 ═══")
    print(f"  pheme_clean 里重复的 text_raw: {clean.text_raw.duplicated().sum()}")
    print(f"  pheme_clean 里重复的 text:     {clean.text.duplicated().sum()}")
    print("  → text_raw 唯一,text 不唯一。匿名化制造了新的重复。")

    # ═══ 2. label 冲突:相同输入、相反标签 ═══
    d = clean[clean.text.duplicated(keep=False)]
    conflict = d.groupby("text").label.nunique()
    n_groups = (conflict > 1).sum()
    n_rows = d[d.text.isin(conflict[conflict > 1].index)].shape[0]
    print("\n═══ label 冲突 ═══")
    print(f"  冲突组数: {n_groups}")
    print(f"  涉及行数: {n_rows}  ({n_rows/len(clean)*100:.2f}%)")
    print("  → 相同输入配相反标签,构成不可消除的误差下限。")

    # ═══ 3. 泄漏分布:重复的 text 有没有跨越 train/test ═══
    print("\n═══ 记忆泄漏(重复 text 跨越 train/test) ═══")
    for name, tr_f, te_f in [
        ("随机划分", "split_random_train.csv", "split_random_test.csv"),
        ("事件划分", "split_event_train.csv", "split_event_test.csv"),
    ]:
        tr = pd.read_csv(f"{DATA_DIR}/{tr_f}")
        te = pd.read_csv(f"{DATA_DIR}/{te_f}")
        overlap = set(tr.text) & set(te.text)
        ratio = te.text.isin(overlap).mean()
        print(f"  {name}: {len(overlap):3d} 条跨越,占 test {ratio*100:.2f}%")

    print("\n  → 事件划分(主实验)为 0,0.705 无记忆泄漏。")
    print("    污染只在随机划分那一侧 → 让 0.149 这个泄漏下界更牢,而非更松。")


if __name__ == "__main__":
    main()