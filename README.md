# PHEME Rumour Detection: Baseline and Competing-Wisdom Augmentation

## Key Result

This project has two stages. **Stage 1** establishes a rigorous baseline
(TF-IDF vs BERT, random vs event-based splits) and, through a two-way error
analysis, shows that both models learn **surface features** — memorised
vocabulary for TF-IDF, emotional style for BERT — rather than whether a claim is
actually verified.

**Stage 2** tests whether a competing-wisdom structure — inspired by L-Defense
(Wang et al., *Explainable Fake News Detection With Large Language Model via
Defense Among Competing Wisdom*, WWW 2024) — can fix this. An LLM generates
opposing "rumour" and "non-rumour" arguments for each tweet, and BERT is trained
on tweet + arguments. The result is a **pre-registered null**: adding arguments
has no detectable effect on macro-F1 (0.705 → 0.705, paired mean diff ≈ 0), and a
one-sided ablation shows removing an entire side changes nothing either. The
arguments carry no usable signal for bert-base.

The likely reason is a step this project drops from L-Defense. L-Defense first
runs an **evidence extractor** (RoBERTa-base) that splits each case into
supporting and refuting evidence, and only then has the arguer write on top of
that evidence. This project has no such step — PHEME provides no paired articles
to retrieve from — so the arguer works from the bare tweet, with nothing external
to reason over. The arguments may therefore be uninformative *by construction*,
regardless of who reads them.

**Conclusion:** the competing-wisdom structure, as applied here, does not help
bert-base — most plausibly because it is missing L-Defense's evidence-retrieval
step, so the arguments have no external grounding to convey. Notably, L-Defense's
own judge is **not an LLM but a fine-tuned RoBERTa-large** — an encoder in the
same family as bert-base — so "the judge is too weak" is not a sufficient
explanation on its own. (A residual caveat: RoBERTa-large is larger than
bert-base, so judge capacity cannot be *completely* ruled out as a secondary
factor; but it cannot be the whole story, since the successful judge is still a
small encoder, not an LLM.) The negative result thus points at the dropped
evidence step as the main missing ingredient — a reverse corroboration of why the
original method is built the way it is.

Full experimental reasoning, thresholds (fixed in advance), and every design
decision are recorded in [`docs/methodology.md`](docs/methodology.md).

## Background

A rumour is information that spreads online while still unverified at the time —
it may later prove true or false. Because rumours often spread faster than they
can be verified, detecting them early is important yet difficult. This project
uses the PHEME dataset, which contains tweets from nine real breaking-news events
labelled as rumour or non-rumour. It first cleans the raw data, then trains
TF-IDF and BERT models, and — through random vs. event-based splits — honestly
assesses how well each generalises to unseen events, including where they fail.

## Dataset

PHEME organises nine real events as nested JSON files. Under each event, tweets
are split into `rumours/` and `non-rumours/` sub-folders, so the label comes from
the folder structure rather than a separate label file. Each tweet folder
contains the source tweet, reactions, and annotations; this project uses only the
source-tweet text. I traverse all nine events, extract the `text` field from each
source tweet's JSON, label it by its folder (rumour = 1, non-rumour = 0), record
its source event, and aggregate everything into 6,425 labelled tweets stored as a
CSV.

## Data Processing

**Deduplication.** I deduplicate first, to prevent the same tweet from landing in
both the training and test sets — which would leak information and inflate scores
through "cheating." I also found one tweet labelled as *both* rumour and
non-rumour: a genuine annotation conflict that would feed the model contradictory
signals and whose true label is unreliable. I therefore removed both of its
copies. This reduced the dataset from 6,425 to 6,406 tweets.

**URLs.** URLs are replaced with a `[URL]` token. The specific link adds little
for rumour detection and is mostly noise, but the fact that a tweet *contains* a
link may be informative — so I keep a placeholder rather than deleting it
outright.

**Mentions.** Mentions (`@username`) are collapsed into a single `@USER` token.
Thousands of distinct usernames are sparse noise to the model; unifying them lets
it learn the signal "a mention exists" instead of memorising who, reducing noise
and overfitting.

**Hashtags.** I strip only the `#` symbol and keep the word (`#Germanwings` →
`Germanwings`). Otherwise TF-IDF would treat `#Germanwings` and the plain word
`Germanwings` as two separate tokens, diluting the signal for the same concept.

The original text is preserved in a `text_raw` column so that cleaning can always
be inspected and traced.

**A later correction — and why `text_raw` earned its keep.** The deduplication
above operated on `text_raw`. During Stage 2 I found that the anonymisation just
described (@USER / [URL]) has a side effect: by erasing the very usernames and
links that distinguished some tweets, it collapses originally-distinct tweets into
identical `text` strings. This created 117 duplicate `text` values — invisible to
the `text_raw` dedup, since in that column they were still distinct — including 6
groups (16 rows) where the duplicates carry conflicting labels. Keeping `text_raw`
is what made the diagnosis possible. Crucially, none of these duplicates cross the
train/test boundary of the **event split**: because that split keeps every event
whole on one side, same-event duplicates always land together and never straddle
the boundary. The event split (used for every main result) is therefore immune to
this leakage; the contamination exists only on the random split, where it pushes
scores up in the same direction as event leakage and so makes the 0.149 gap a
firmer *lower* bound. The label conflicts are a separate matter: unlike the duplicate leakage, they
are not neutralised by the event split — those rows sit in whichever event
they belong to and add a small irreducible error to both conditions equally.
What makes them harmless to the main comparison is the paired design: identical
rows contribute identical error to baseline and augmented, so they cancel in the
per-seed difference. Full analysis and the decision to keep these rows in
[`docs/methodology.md`](docs/methodology.md).

## Exploratory Analysis

Three findings from the cleaned data shaped the later design:

**Class imbalance.** Non-rumours outnumber rumours about 1.7:1 (4,012 vs 2,394).
A model that simply always predicts "non-rumour" would still reach over 60%
accuracy, so I evaluate with macro-F1 and rumour recall instead of accuracy.

**Highly uneven event sizes.** Events range from 14 tweets (ebola-essien) to over
2,000 (charliehebdo). Very small events are unreliable as a test set — a score
computed over 14 examples is statistically meaningless — so such events must be
avoided when designing the event-based split.

**Rumour ratio varies dramatically across events**, from 22% to 100%. This is the
key finding: a model could "cheat" by recognising which event a tweet belongs to
and guessing that event's dominant label — a form of event-level data leakage.
This motivated evaluating under both a random split and an event-based split.

![Rumour ratio per event](results/rumour_ratio.png)

## Experiments

I fine-tune BERT (`bert-base-uncased`, 3 epochs, 5 seeds) and train a TF-IDF +
Logistic Regression baseline, evaluating both under the random and event-based
splits.

| Model  | Split       | Macro-F1        | Rumour Recall   |
|--------|-------------|-----------------|-----------------|
| TF-IDF | Random      | 0.848           | 0.808           |
| BERT   | Random      | 0.854 ± 0.007   | 0.796 ± 0.012   |
| TF-IDF | Event-based | 0.665           | 0.503           |
| BERT   | Event-based | 0.705 ± 0.010   | 0.573 ± 0.031   |

BERT scores are the mean ± std over 5 seeds; TF-IDF is a single run (no random
classification head to seed). The 5-seed protocol came out of the reproducibility
fix described below — the original single-run BERT numbers were not reproducible
and have been discarded.

### Reproducibility fix

The initial BERT setup called `set_seed()` *after* `from_pretrained()`. But
`from_pretrained(num_labels=2)` randomly initialises the classification head at
that point, before the seed takes effect — so the head's initialisation was
uncontrolled and the baseline could not be reproduced run to run. Moving
`set_seed()` before model creation fixed this; all reported BERT numbers are now
reproducible across the 5 seeds. See [`docs/methodology.md`](docs/methodology.md).

### Observations

**The two models are close on the random split** (F1 0.848 vs 0.854). This split
lets TF-IDF "cheat" by memorising rumour-related words seen in training; since
tweets from the same events reappear at test time, memorisation is enough, and
BERT's semantic understanding gives little advantage.

**Both drop sharply on the event-based split** (F1 falls to 0.66–0.71). With the
test set drawn from entirely unseen events, this confirms that the event leakage
flagged during EDA is real and substantial.

**But BERT drops less and recovers more recall** (TF-IDF 0.503 → BERT 0.573).
Facing unseen vocabulary in new events, TF-IDF can only match words it has already
seen and misses many rumours, whereas BERT's subword tokenisation and semantic
understanding let it recognise rumours even from words it has not seen before.

## Error Analysis

I ran a two-way error analysis on the BERT model under the event-based split, and
each error type turned out to concentrate on a single event:

**False negatives (rumours predicted as non-rumours) fall almost entirely in
putinmissing.** These political rumours read like serious news — calm wording,
quoted experts, attached media links — and lack the sensational, urgent cues the
model learned from other events, so they slip through as credible reports.

**False positives (non-rumours predicted as rumours) fall almost entirely in
ottawashooting.** These are actually real breaking-news tweets, but their
aggressive, panicked, urgent wording (e.g. BREAKING, lockdown warnings) matches
the "rumour tone" the model learned, so they are misclassified.

Both directions point to the same conclusion: **the model judges by emotional
style (sensational vs. calm) rather than by whether the content is actually
verified.** In other words, it has learned "sensational style" as a surface proxy
instead of the essence of a rumour (unverified information) — which also explains
why it fails to generalise to events with a different style.

## Key Findings

On the random split, both TF-IDF and BERT reach solid scores — but largely by
exploiting event leakage, having seen tweets from the same events during training.
Under the event-based split, facing entirely unseen events, both drop sharply,
showing that cross-event generalisation is the real challenge.

More deeply, the two models fail in different ways but for the same reason:
**both learn surface features rather than the essence of a rumour.** TF-IDF judges
by memorising rumour-related words from the training set and breaks on unseen
vocabulary; BERT, though it understands semantics, still largely judges by the
emotional style of the wording and is fooled by rumours written in an unfamiliar
tone. A rumour is fundamentally *unverified information*, and judging it should
rest on verifying the content — which neither model truly does. This is the root
cause of their poor generalisation.

This points toward a more promising direction: having models retrieve evidence and
verify claims rather than guessing from surface cues in the text.

## Stage 2: Competing-Wisdom Augmentation

### Motivation

Stage 1 showed BERT judges by emotional style, not by whether a claim is verified.
Competing wisdom is an attempt to force the model to reason: instead of the tweet
alone, feed it two opposing arguments — one defending "this is a rumour," one
defending "this is not" — and let it weigh them.

### Generating the arguments

An LLM (MiniMax-M3) plays a defense lawyer: given a tweet and a target verdict, it
writes the strongest case for that verdict using only clues in the text (named
sources, verifiability, internal consistency, emotional exaggeration). Two hard
constraints protect the experiment:

- **The gold label never enters the prompt.** The generator argues both sides
  blind. A label reaching it would be leakage — the model would recall the answer
  and dress it up as reasoning.
- **No outside knowledge.** PHEME events (2014–15) predate the LLM's training
  data, so allowing external facts would let it retrieve the answer instead of
  analysing the text. Arguments are grounded strictly in the tweet.

This produced 6,406 argument pairs. The input BERT sees is:
`{tweet} [SEP] RUMOUR: {arg} [SEP] NOT RUMOUR: {arg}`.

### Result: a pre-registered null

The decision threshold (0.013, derived from baseline noise) and the primary
metric (macro-F1) were **fixed and committed to git before running** — so the
result cannot be a moved goalpost. All comparisons use paired per-seed differences
(baseline and augmented share the same 5 seeds).

| Condition | Input | Macro-F1 (event split, 5 seeds) |
|---|---|---|
| Baseline | tweet only | 0.705 ± 0.010 |
| Two-sided | tweet + both arguments | 0.705 ± 0.028 |
| One-sided (ablation) | tweet + rumour argument only | 0.706 ± 0.026 |

**The three conditions are indistinguishable** — every paired difference sits well
inside the ±0.013 threshold. Adding two opposing arguments does not help; removing
one entire side (the ablation) changes nothing either.

### Why: the missing evidence-retrieval step is the likely cause

The ablation tells us the argument *content* does not matter: deleting an entire
side changes nothing, and even one side is no better than no argument at all. The
question is *why* the content is inert. Comparing this setup to L-Defense points
to one main cause.

L-Defense runs an **evidence extractor** (RoBERTa-base) that splits each case into
supporting and refuting evidence *before* the arguer writes, so its arguments are
grounded in retrieved facts. This project drops that step — PHEME has no paired
articles to retrieve from — so the arguer reasons from the bare tweet alone. With
nothing external to draw on, the arguments may carry no real signal **by
construction**: there is simply nothing informative to encode, so no reader could
extract anything from them.

A weaker alternative — that bert-base is too small to *read* the arguments even if
they were informative — is largely ruled out by L-Defense itself: its judge is a
fine-tuned **RoBERTa-large**, a small encoder in the same family as bert-base, not
an LLM. A same-class model can evidently use these arguments when they are
grounded. (RoBERTa-large is somewhat larger than bert-base, so reader capacity
cannot be *completely* excluded as a minor factor — but it cannot be the main
story, since the working judge is still a small encoder.)

So the evidence points at the dropped retrieval step, not the reader. Confirming
this requires re-running with retrieved evidence where available (see Future Work).

One honest detail: the augmented conditions are *noisier* than baseline (σ rises
from 0.010 to ~0.027). Adding LLM text doesn't move the mean but does inject
variance — for bert-base, it is essentially noise.

### Takeaway

The competing-wisdom structure, as applied here, does not help bert-base — most
plausibly because this project strips out L-Defense's evidence-retrieval step, so
the arguments have no grounded facts to convey. Tellingly, L-Defense's own judge
is a small encoder (RoBERTa-large), not an LLM, so the failure is unlikely to be
about the reader's strength. The negative result therefore points back at the one
part of the original pipeline this simplification removed — a reverse
corroboration of why that part is there. Full reasoning in
[`docs/methodology.md`](docs/methodology.md).

## Repository Structure

**Stage 1 — baseline**

- `prepare_data.py` — Load raw PHEME JSON, deduplicate and clean → `pheme_clean.csv`
- `eda.py` — Exploratory analysis; saves plots to `results/`
- `split_data.py` — Random stratified split and event-based split (seeded)
- `train_tfidf_compare.py` — TF-IDF + Logistic Regression baseline (both splits)
- `train_bert.py` — BERT fine-tuning, 5 seeds (run on Colab GPU)
- `error_analysis.py` — Two-way error analysis (false negatives / false positives)

**Stage 2 — competing-wisdom augmentation**

- `generate_explanations.py` — Generate opposing arguments via LLM (resumable, 6,406 pairs)
- `check_length.py` — Token-length measurement; owns `combine()`, the single source of truth for the input format
- `check_duplicates.py` — Diagnostic for anonymisation-induced duplicates and label conflicts
- `train_bert_augmented.py` — BERT on tweet + both arguments (5 seeds)
- `train_bert_ablation.py` — One-sided ablation: rumour argument only (5 seeds)

**Shared / config**

- `config.py` — Single source of truth for seeds and hyperparameters (frozen before results)
- `docs/methodology.md` — Every design decision, threshold, and result, with rationale
- `results/` — EDA plots and per-experiment JSON results

The raw PHEME data and generated CSVs are not tracked in git (see `.gitignore`);
they are regenerated from the scripts. The one exception is
`data/explanations/competing_wisdom_minimax_m3.jsonl` — the LLM-generated
arguments, which are an irreproducible API output and are tracked so the augmented
experiment can be replicated.

## How to Run

Environment is managed with [uv](https://github.com/astral-sh/uv).

**1. Data preparation.** Download the PHEME dataset ("PHEME dataset for Rumour
Detection and Veracity Classification", figshare) into `data/`, then:

```bash
uv run python prepare_data.py    # → data/pheme_clean.csv
uv run python eda.py             # → plots in results/
uv run python split_data.py      # → splits in data/
```

**2. TF-IDF baseline** (runs locally):

```bash
uv run python train_tfidf_compare.py
```

**3. BERT** (Google Colab with GPU): upload the split CSVs to Google Drive, mount
Drive, `pip install transformers datasets`, then run `train_bert.py`; run
`error_analysis.py` on the trained event-split model to reproduce the error
analysis.

**4. Stage 2** (Colab): the LLM arguments are already generated and tracked
(`data/explanations/competing_wisdom_minimax_m3.jsonl`). Run `train_bert_augmented.py`
for the two-sided condition and `train_bert_ablation.py` for the one-sided ablation.

## Future Work

- **Add back the evidence-retrieval step.** The Stage 2 null most likely comes
  from arguing over the bare tweet with no retrieved evidence — the one step this
  project drops from L-Defense. Re-running with an evidence source (where PHEME or
  an external corpus allows) would test directly whether grounded arguments become
  useful, confirming or refuting the diagnosis above.
- **Improve cross-event generalisation**, since both models rely on surface
  features (vocabulary or emotional style) rather than verifying content.
  Approaches that retrieve external evidence or reason over claims are a promising
  direction.
- **Address event-specific leakage** — e.g. numeric hashtags like `#4U9525` act as
  event identifiers and may let models shortcut rather than learn genuine rumour
  cues.
- **Select the best epoch using a validation set** for the event-based split, as
  the models showed overfitting (rising validation loss) after the first epoch.
- **Clean residual noise** in the dataset (a few off-topic tweets, e.g. sports
  posts, appear under some events).

## Reference

This project's Stage 2 is inspired by (not a reproduction of):

> Bo Wang, Jing Ma, Hongzhan Lin, Zhiwei Yang, Ruichao Yang, Yuan Tian, Yi Chang.
> **Explainable Fake News Detection With Large Language Model via Defense Among
> Competing Wisdom.** WWW 2024, Singapore.
> [[Paper](https://dl.acm.org/doi/10.1145/3589334.3645471)]
> [[arXiv](https://arxiv.org/abs/2405.03371)]
> [[Code](https://github.com/wangbo9719/L-Defense_EFND)]

**How this project differs (§13 red line):** L-Defense has three components — an
evidence extractor (RoBERTa-base), an LLM arguer, and a fine-tuned RoBERTa-large
judge. This project keeps the competing-wisdom structure and the two-sided
argumentation but **drops the evidence-retrieval step** — PHEME has no paired news
articles to retrieve from, so the LLM argues directly from the tweet text. It is
therefore an *inspired-by simplification*, not a reproduction. (Note the judge in
the original is itself a small encoder, not an LLM — which is why the negative
result here points at the missing evidence step rather than at model size.)

The augmentation ultimately did not help bert-base (see Key Result), which is
itself the finding.