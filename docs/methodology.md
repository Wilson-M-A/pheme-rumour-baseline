# Methodology Decisions

> This document records the rationale behind each experimental design decision.
> All "fixed-in-advance" commitments are timestamped by this file's git history.

## Decision Threshold (fixed before running the augmented experiment)

Baseline (event split, 5 seeds): macro_f1 = 0.705 ± 0.010

### Detectable threshold
Single-run σ ≈ 0.010; σ of the 5-seed mean ≈ 0.0045; σ of the difference
between two means ≈ 0.0063. The smallest difference this experiment can
resolve is therefore about **0.013**.

Below this value, the correct statement is "no significant difference was
observed at this experiment's precision," not "there is no difference." The
former is a claim about the instrument; the latter is a claim about the world.
This experiment only earns the former.

### Decision criterion
**Detectable threshold = threshold-worth-caring-about = 0.013.**

Rationale: this experiment asks whether the competing-argument *structure*
works, not whether the method is worth deploying. The former only requires a
reliably detectable effect; cost arguments (generating 6,406 arguments costs
~$5 and ~4 hours) belong to the latter and are out of scope here.

Cost of this choice: a result of +0.015 would only support "a small but real
effect was detected," **not** "this is a good method" — the evidence for the
latter is not collected here.

### Primary metric
**Primary metric: macro_f1, threshold 0.013.**

rumour_recall is an exploratory observation: it may be reported but cannot be
used to claim the method works. Rationale: with enough metrics, one will always
move (the garden of forking paths). If macro_f1 is unchanged while recall rises,
that is a lead worth following, not a conclusion.

### Analysis method
Baseline and augmented runs share the same seed set (paired design), so the
comparison is over the distribution of per-seed differences d_i = aug_i − base_i,
not over the two means. Pairing removes the "some seeds simply run better"
component of the noise.

## Anonymization-Induced Duplicates and Label Conflicts (not covered by v1 dedup)

Deduplication of `pheme_clean.csv` was performed on `text_raw` (0 duplicates),
but the model is trained on `text`. Anonymization (@USER / [URL]) collapses
distinct tweets into identical strings, producing **117 duplicate `text` values**,
of which **6 groups (16 rows)** carry conflicting labels — identical input,
opposite label — forming an irreducible error floor (0.25% of the data).

Leakage distribution: the random split has **30 `text` values crossing
train/test** (3.3% of test; estimated impact <0.005, below the detectable
threshold of 0.013); the **event split has 0**. The main experiment's 0.705 is
therefore free of memorization leakage — the contamination exists only on the
0.854 (random-split) side, which makes the 0.149 leakage gap a firmer lower
bound, not a looser one.

Decision: **keep them.** These rows are byte-for-byte identical in the baseline
and augmented runs (arguments are a function of `text` alone; labels never
entered the generation prompt), contributing zero to the main comparison — the
difference between the two conditions. The re-run cost (45 min) buys no precision
relevant to the research question.

How it was found: the `assert len(merged) == len(df)` during the merge fired
immediately, catching 5282 → 5528.