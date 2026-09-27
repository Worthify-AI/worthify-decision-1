# Four-epoch BANKING77 LoRA follow-up

The [predeclared extended protocol](FOUNDATION_BANKING77_EXTENDED_LORA.md)
ran four fresh LoRA adapters for 388 optimizer updates each: seeds 42 and 43
on original Gemma 4 12B and on the private Decision-1 preview. The protocol
reused the same 1,540 training rows, 365 validation rows, 3,080 official test
rows, deterministic 16-option routing conversion, model/source revisions,
LoRA configuration, optimizer, warmup, batch sizes, and numeric policy as the
[earlier two-epoch study](FOUNDATION_BANKING77_TRANSFER_EVAL.md). It is an
exploratory extension chosen after viewing the earlier result, not an
independent confirmatory test.

Validation-only selection chose seed 42 for each base. The original-base
adapter at update 350 scored **93.54%** on the sealed test; the Decision-1
adapter at update 325 scored **93.21%**. The tuned-minus-original difference
was **−0.32 percentage points** (10 fewer correct of 3,080). The paired table
has 2,821 both correct, 149 both wrong, 60 original only correct, and 50
Decision-1 only correct. The 77-intent cluster bootstrap interval for the
selected difference was **−0.94 to +0.29 points**. Seed 43 likewise favored
the original base: 93.77% versus 92.66%, a **−1.10-point** tuned-minus-original
difference. Excluding the 28 previously flagged full-weight v9 exposure rows
left a **−0.36-point** selected-pair difference. The predeclared task-specific
improvement gate therefore failed.

The [locked validation chart](assets/decision-1/banking77-validation-0-388-20260927-v1.png)
shows both adapter seeds and the mean for each base on the same fixed update
grid. The Decision-1 curves begin lower and catch up during the later updates.
Both seed pairs have higher **baseline-adjusted** validation-accuracy area for
Decision-1, and both have higher area over updates 194–388. Original Gemma has
higher **absolute** area over updates 0–388 in both seed pairs. At update 388,
the validation accuracies were 94.79%/94.52% for original seeds 42/43 and
95.34%/94.25% for Decision-1 seeds 42/43. These validation trends do not
override the sealed-test result.

The extension started from fresh adapters because the earlier checkpoints
did not preserve AdamW optimizer or RNG state. Initial LoRA tensor hashes
match between bases within each seed. In the old-versus-new reproducibility
check, adapter bytes match through update 75 and logged losses and gradient
norms match through update 96 for all four run pairs. The update-97 loss also
matches, but its gradient norm differs on the four-row epoch-end batch; model
weights and later validation curves then diverge. No algorithmic change was
found in the trainer beyond epoch count and protocol/schema identifiers, and
the source of that gradient difference was not isolated. The first 194
updates should not be read as a byte-exact continuation of the earlier run.

The [public report](../results/worthify/foundation-banking77-extended-lora-20260927-v1/report.json),
[paired correctness rows](../results/worthify/foundation-banking77-extended-lora-20260927-v1/paired-rows.jsonl),
and [raw aggregate](../results/raw/foundation-banking77-extended-lora-20260927-v1.json)
contain the exact metrics, selected steps, model hashes, validation curves,
paired test outcomes, and SHA-256 provenance. The evidence excludes utterance
text, prompts, option logits, model weights, and private local paths. This is
16-option routing, not standard 77-way BANKING77 classification. The task was
selected after a negative CTU-13 result; lexical decontamination cannot prove
semantic novelty; the Decision-1 base is one provisional full-weight seed.
