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

## Result: No Detectable Effect on macro_f1

Augmented (event split, 5 seeds): macro_f1 = 0.705 ± 0.028

Paired differences d_i = aug_i − base_i (same seed set):

| seed | base  | aug   | diff   |
|------|-------|-------|--------|
| 42   | 0.716 | 0.704 | −0.012 |
| 43   | 0.699 | 0.713 | +0.014 |
| 44   | 0.691 | 0.745 | +0.055 |
| 45   | 0.711 | 0.693 | −0.018 |
| 46   | 0.706 | 0.668 | −0.038 |

mean d = −0.000, σ_d = 0.036, σ of mean d = 0.016.

**At this experiment's precision, no reliable effect of competing wisdom on
macro_f1 was detected** (|mean d| < detectable threshold 0.013). This is a
statement about the instrument, not the world: the experiment cannot rule out
a true effect smaller than ~0.013.

Two observations that make the null informative rather than empty:

1. **The near-zero mean is cancellation, not stability.** Per-seed differences
   range from −0.038 to +0.055 — the same treatment helps some runs and hurts
   others, suggesting competing wisdom behaves more like a high-variance
   perturbation than a stable offset (though with n=5, this is a hypothesis,
   not an established property).

   Note on resolving power: the 0.013 threshold was derived from the baseline's
   noise (σ ≈ 0.010), but the augmented condition turned out noisier
   (σ_d = 0.036), so this experiment's *actual* resolving power here is coarser
   (~0.032). The mean shift (−0.000) sits far inside even this wider band, so the
   null holds under either figure — and the increased variance is itself a
   finding, not a nuisance.

2. **recall did not save the result.** rumour_recall moved 0.573 → 0.580,
   well within its own σ (0.065). Per the pre-registered primary metric, this
   is noise, not evidence — recorded, not claimed.

Open question (to be addressed by the one-sided ablation): since the input is
~90% LLM-generated text and only ~10% tweet, does the model read argument
*style* rather than *content* — reproducing the same shortcut-learning failure
seen in the baseline, one level up?

> **Update (after ablation + reading L-Defense):** the ablation reframed this —
> see "Answer to the open question" below. The relevant difference turned out to
> be the missing evidence-retrieval step, not reader strength.

## One-Sided Ablation: Pre-Registered Interpretation (written before running)

Setup: identical to the augmented experiment in every respect (same seeds,
hyperparameters, MAX_LENGTH=512, dynamic padding, event split, paired analysis)
except one variable — the NOT RUMOUR slot keeps its label but is emptied of
content. Input format: `{text} [SEP] RUMOUR: {r_arg} [SEP] NOT RUMOUR:`.

This isolates a single question: **does the model use the argument's *content*,
or merely the *presence and position* of a second side?** Same primary metric
and threshold as before (macro_f1, detectable threshold 0.013), compared both
against baseline and against the two-sided augmented condition.

Two outcomes, interpretations fixed in advance:

- **One-sided ≈ 0.705 (≈ two-sided ≈ baseline):** the model is not reading
  argument content — it responds only to position/presence, or not at all.
  This would mean the competing structure is not being exploited, and the
  baseline's style-shortcut failure has simply reappeared one layer up, on the
  LLM-generated text. It would give the two-sided null a complete explanation:
  the structure adds nothing because the model never uses either side's content.

- **One-sided clearly < 0.705:** removing the non-rumour side's content
  measurably hurts the model, so the opposing content *is* being used. The
  two-sided null would then read not as "the structure is useless" but as
  "a real effect exists but is drowned out" — e.g. by the ~90% LLM-generated
  text acting as noise.

No third interpretation will be invented after seeing the number.

> **Retrospective note:** after running the ablation I checked L-Defense's actual
> architecture and found its judge is RoBERTa-large (a small encoder, not an LLM).
> This weakens interpretation A ("not reading") — a same-class model can use these
> arguments when they are evidence-grounded — and points to the dropped
> evidence-retrieval step as the more likely cause. The interpretations above are
> left unedited as the genuine pre-registration; this note records what changed.

## Ablation Result: The Structure Is Not Being Used

One-sided (event split, 5 seeds): macro_f1 = 0.706 ± 0.026

Paired differences (same seed set):

| comparison | mean d | σ_d | verdict |
|---|---|---|---|
| one-sided − baseline | +0.001 | 0.025 | within threshold |
| one-sided − two-sided | +0.001 | 0.048 | within threshold |

All three conditions — raw tweet, tweet + one argument, tweet + two opposing
arguments — are **indistinguishable in macro_f1** (0.705 / 0.705 / 0.706).

Removing a full side of argument content changes nothing (one-sided = two-sided),
and a single side is no better than no argument at all (one-sided = baseline). The
argument content, in other words, is inert: with zero, one, or two sides, the
model decides the same way.

Note on variance: the one-sided vs two-sided comparison (σ_d = 0.048) is nearly
twice as noisy as one-sided vs baseline (σ_d = 0.025). Adding the second side
does not raise the mean but does inject variance — consistent with the augmented
result. For bert-base, the LLM-generated text acts as noise, and more of it
means more instability.

### Answer to the open question
The augmented input is ~90% LLM-generated argumentation, yet removing an entire
side of it changes nothing (see the ablation above). The most likely reason is
**not** that bert-base is too weak to read the arguments — L-Defense's own judge
is a fine-tuned **RoBERTa-large**, a small encoder in the same family as
bert-base, not an LLM, and it uses such arguments successfully. The difference
lies elsewhere: L-Defense first runs an **evidence extractor** (RoBERTa-base) so
its arguer writes over retrieved facts, whereas this project drops that step
(PHEME has no paired articles) and the arguer reasons from the bare tweet alone.

The arguments are therefore likely uninformative *by construction* — with no
external evidence to ground them, the arguer only re-describes the tweet, so no
reader (small or large) can extract new signal. The null is a reverse
corroboration of why L-Defense includes evidence retrieval, not of any claim
about judge size.

(Residual caveat: RoBERTa-large is somewhat larger than bert-base, so reader
capacity cannot be *completely* excluded as a minor factor — but it cannot be the
main cause, since the working judge is itself a small encoder. Confirming the
evidence-retrieval hypothesis requires re-running with retrieved evidence; see the
project README's Future Work.)