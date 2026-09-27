# Exploratory matched LoRA comparison on BANKING77 routing

This is a **second task chosen after the CTU-13 matched comparison showed a
regression** for the Worthify full-weight base. It is a domain-aligned follow-up,
not evidence that the task was chosen blind to earlier results. This protocol
must be frozen before any BANKING77 LoRA training, validation selection, or
official-test scoring in this comparison. The official test is used by a
separate, label-blind lexical decontamination audit before training; that
audit may report only counts and hashes, never test text or performance.
Both outcomes, including the earlier CTU-13 negative result, belong in any
public claim or model-card discussion.

## Fixed arms, data, and training budget

- **Original base:** `google/gemma-4-12B-it` at
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- **Tuned base:** the verified private, single-seed full-weight preview
  `Worthify/worthify-jev-gemma4-12b-full-v9-seed42-preview` at
  `8861f801f2b4f0b9943438bdcb5500f0c4a48894`. Its BF16 export
  receipt has SHA-256
  `56a1e507c1dded0cc6eba54713d8e4f513b38cc0648d74fa7ea4fb8dac32941f`.
  This is not a selected two-seed model release.
- **Source and task:** official publisher BANKING77 train and test files and
  taxonomy at revision `57ec275d8078af65b7731c2a98be812d844a6d6b`,
  acquired only through `benchmarks/fetch_sources.py` with independent
  SHA-256 discovery and refetch. Credit Casanueva et al. (2020) and the
  publisher's CC BY 4.0 license. Every utterance is routed among a
  deterministic set of **16** of the 77 original intent labels, including
  its gold label. This is not a standard 77-way BANKING77 score.
- **Split and decontamination:** keep the publisher's entire official test
  split sealed for model scoring. Use the existing publisher-train-derived
  candidate's disjoint whole-group train/validation assignment. Before
  subsampling, reject whole groups with a normalized exact or one-token
  deletion overlap against frozen evaluation records, other relevant
  full-weight training/validation records, or official BANKING77 test
  utterances. The independent audit may use test *utterance signatures*
  solely to exclude contaminated training/validation groups; it may not
  expose test labels or text to model fitting or checkpoint selection.
  Select at most 20 train rows and at most 5 validation rows per original
  intent by a fixed hash, retaining complete source groups and never moving
  rows between splits. Record every final row, group, intent count, exclusion
  count, source hash, and split hash in a create-only private manifest before
  the first GPU run. Require all 77 intents in train and report validation
  coverage. A failed leakage or coverage gate stops the comparison.
- **Four matched runs:** each base receives a fresh LoRA at seeds 42 and 43.
  For a given seed, require the initial LoRA tensor hash to match between
  bases. Rank 16, alpha 32, dropout 0.05, q/k/v/o modules, NF4 base weights,
  BF16 computation, FP32 final softcap at 30, identical option-logit
  cross-entropy loss and prompt, identical seed-determined training order,
  global batch 16 with microbatch 8, AdamW learning rate `2e-5` with 50-step
  linear warmup, gradient clipping at 1, and two complete epochs. Expose
  exactly one CUDA GPU per scorer process. Report total updates from the
  frozen train count. The prepared split and protocol hashes must match all
  four receipts.

## Selection and predeclared outcomes

Evaluate validation before the first update and at every 25 updates, both
epoch ends, and the final update. If these coincide, evaluate once. Select a
checkpoint **only after all distinct training rows have been seen** and rank
by validation accuracy, then macro-F1 over the 77 original intents (absent
validation labels receive zero), then lower validation cross-entropy; take
the earliest update on an exact tie. Select one seed per base with that same
ranking and lower seed on an exact tie. Lock all four runs, selected adapters,
validation predictions, and hashes before test scoring. A fresh process must
reproduce every selected validation row's option logits and choice to a
`1e-4` logit tolerance before it can open the test split. Score all four
locked adapters on the same official test rows and options. No test result
may change the subset, prompt, hyperparameters, checkpoint, or seed.

The **primary endpoint** is tuned minus original test accuracy for the two
validation-selected LoRAs on the official test, with both scores and the
paired correctness table. Secondary endpoints are the two seed-pair test
accuracy differences, the 77-intent macro-F1, per-intent accuracy, step-zero
validation accuracy, fixed-budget validation accuracy, and the normalized
trapezoidal area under the validation-accuracy curve from step zero through
the final update. Also report the area after subtracting each arm's own
step-zero accuracy. Report the first checked update reaching validation
accuracy 0.80; if this happens at zero, report zero but do not call it faster
learning. Report wall time separately from update efficiency.

For descriptive uncertainty, resample the 77 test intent groups with
replacement 10,000 times using Python seed 20260926; preserve all rows of
each sampled group and recompute paired accuracy differences. Report the
2.5th and 97.5th percentiles. A public **task-specific improvement** claim
requires the selected pair and both predeclared seed pairs to favor the
tuned base, a positive lower percentile bound for the selected pair, and
positive selected-pair and per-seed differences in the exposure-excluded
sensitivity analysis defined below.
Use **"faster learning"** only if both seed pairs favor the tuned base on
baseline-adjusted validation-accuracy area; a high step-zero score alone
does not meet this criterion. If these gates fail, state the negative or
mixed result without substituting a more favorable metric.

This is an exploratory follow-up on one dataset and one 16-option conversion.
The base already received other intent and reasoning training, including
CLINC150, so this cannot establish de novo learning or broad LoRA superiority.
Lexical decontamination cannot prove semantic novelty. Grouped test
resampling describes variation across these 77 BANKING77 intents; it does
not estimate variation across future tasks or base-training seeds.

The pre-training lexical audit also found a small number of official test
utterances that overlap full-weight v9 train, validation, or test records.
Keep the full 3,080-row official test as the primary endpoint. As a declared
**sensitivity analysis**, recompute selected-pair and per-seed accuracy
differences after excluding the *union* of those audit-flagged test row IDs.
Freeze that ID set and its hash before GPU training. Report the overlap
counts by v9 split and in union, and do not present the remaining subset as
proven semantically unseen. A favorable primary result that disappears in
this sensitivity analysis cannot justify a robust transfer claim.
