"""
Measure the real BERT token length of the augmented input.

WHY: the augmented input is `tweet + rumour argument + not-rumour argument`.
bert-base-uncased caps at 512 tokens (its position embeddings only go that far),
and truncation always cuts from the END -- which is the NOT-A-RUMOUR side.
So an over-length sample doesn't just lose "some text": it loses one whole side
of the debate, silently collapsing the competing-wisdom structure into a
one-sided input. That is exactly the thing this experiment is testing, so we
measure before training rather than guess.

Read the tail, not the mean: an average of 270 tells you nothing if 5% sit above
512. `max` and the over-limit share are what decide max_length.

Usage:
  uv add transformers numpy
  uv run python check_length.py
"""

import json
import numpy as np
from transformers import AutoTokenizer

JSONL = "data/explanations/competing_wisdom_minimax_m3.jsonl"
MODEL = "bert-base-uncased"


def combine(rec):
    """The exact string BERT will see.

    SINGLE SOURCE OF TRUTH: train_bert_augmented.py must import this function
    rather than re-typing the format. If the two ever drift apart, this whole
    measurement describes a string that never actually gets trained on.

    [SEP] is a real token in BERT's vocab, so it reads as a genuine boundary.
    The RUMOUR: / NOT RUMOUR: prefixes are what tell BERT which side is which --
    token_type_ids can't do it, since BERT only has 2 segments, and we have 3 parts.
    """
    return (f"{rec['text']} "
            f"[SEP] RUMOUR: {rec['rumour_argument']} "
            f"[SEP] NOT RUMOUR: {rec['nonrumour_argument']}")


def load_jsonl(path):
    """One independent JSON object per line -- so json.loads line by line,
    NOT json.load on the whole file. The `if line.strip()` guards against a
    blank or half-written final line from an interrupted run."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def report(name, lens):
    lens = np.array(lens)
    p50, p90, p95, p99 = np.percentile(lens, [50, 90, 95, 99])
    print(f"\n── {name} (n={len(lens)}) ──")
    print(f"  p50 {p50:.0f} | p90 {p90:.0f} | p95 {p95:.0f} | p99 {p99:.0f} | max {lens.max()}")
    return lens


def main():
    recs = load_jsonl(JSONL)
    print(f"loaded {len(recs)} records from {JSONL}")

    tok = AutoTokenizer.from_pretrained(MODEL)

    combined = [combine(r) for r in recs]
    # tok() already inserts [CLS] and [SEP], so len(input_ids) IS what BERT ingests.
    lens = report("combined (tweet + both arguments)", [len(tok(t).input_ids) for t in combined])

    # Component breakdown: shows where the length actually comes from.
    report("tweet only", [len(tok(r["text"]).input_ids) for r in recs])
    report("rumour argument", [len(tok(r["rumour_argument"]).input_ids) for r in recs])
    report("not-rumour argument", [len(tok(r["nonrumour_argument"]).input_ids) for r in recs])

    # ── The actual decision ──
    print("\n" + "=" * 60)
    print("truncation risk by max_length:")
    for L in (128, 256, 384, 512):
        n_over = int((lens > L).sum())
        pct = n_over / len(lens) * 100
        flag = "  <-- baseline used this" if L == 128 else ""
        print(f"  max_length={L:3d} : {n_over:5d} / {len(lens)} truncated ({pct:5.2f}%){flag}")

    print("\nrecommendation:")
    hard_cap = 512   # bert-base-uncased cannot go higher
    for L in (128, 256, 384, 512):
        if (lens > L).sum() == 0:
            print(f"  max_length={L} is the smallest setting with ZERO truncation.")
            print(f"  Attention cost is O(n^2), so smaller is cheaper on the T4 --")
            print(f"  and with dynamic padding (DataCollatorWithPadding) the real cost")
            print(f"  tracks actual length anyway; max_length just sets the ceiling.")
            break
    else:
        over = (lens > hard_cap).mean() * 100
        print(f"  !! {over:.1f}% still exceed {hard_cap}, which is bert-base-uncased's hard limit.")
        print(f"     Options: shorten the prompt (1-2 sentences per side) and regenerate,")
        print(f"     or accept that the NOT-A-RUMOUR side gets cut on those samples.")

    print("\nquote this number in the README, e.g.:")
    print(f'  "measured max = {lens.max()} tokens over {len(lens)} samples -> max_length set accordingly"')


if __name__ == "__main__":
    main()