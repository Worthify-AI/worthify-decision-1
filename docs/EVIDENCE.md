# Evidence and method

The model card reports a single full-weight seed and matched LoRA studies on
two tasks, including a longer BANKING77 follow-up. These files contain prediction IDs, labels and scores; they do not
contain publisher utterances, prompts, source records or model weights.

| Evidence | What it establishes |
|---|---|
| [Adapter-free → task-LoRA rows and report](../results/worthify/foundation-banking77-unadapted-transfer-20260927-v1/) | Same 3,080 BANKING77 16-option test rows before and after the selected task LoRA on each base; paired gains and intent-bootstrap intervals. |
| [Earlier BANKING77 rows and report](../results/worthify/foundation-banking77-transfer-lora-20260926-v1/) | Four fresh adapters; seeds 42/43 on each base; 3,080 transformed official-test rows. |
| [Extended BANKING77 rows and report](../results/worthify/foundation-banking77-extended-lora-20260927-v1/) | Four fresh adapters; 388 updates each; the same 3,080 transformed official-test rows. |
| [CTU-13 rows and report](../results/worthify/foundation-transfer-lora-20260926-v1/) | Four fresh adapters on one separate capture scenario, 407 dependent test rows. |
| [Full-weight held-out predictions](../results/worthify/full-weight-seed42-heldout-20260924/) | Frozen base, earlier task LoRA and full-weight preview on the same 3,200 rows. |

## Before and after task training

The new baseline scored each unadapted base once with the same prompt, 16
options, NF4/BF16 loading, FP32 final softcap and 2,048-token cap as the
four-epoch LoRA comparison. Both unadapted bases reproduced the archived
update-zero validation choices before official-test scoring. The test baseline
was specified **after** the adapter outcomes were known, so it is descriptive,
not a new sealed experiment.

Decision-1 rose from **2,698/3,080 (87.60%)** without a task adapter to
**2,871/3,080 (93.21%)** with its validation-selected seed-42 adapter:
**+5.62 percentage points**. Original Gemma rose from **2,707 (87.89%)** to
**2,881 (93.54%)**: **+5.65 points**. Paired 77-intent bootstrap intervals are
**+3.21 to +8.57** and **+3.28 to +8.44 points**, respectively. Excluding 28
lexical-overlap-flagged rows leaves **+5.67** and **+5.64 points** on 3,052
rows. The [CPU verifier](../benchmarks/verify_banking77_unadapted.py) recomputes
these values from the text-free paired rows and joins the LoRA predictions to
the earlier public study.

"Zero-shot" means no BANKING77 task adapter or prompt examples. The full-weight
training included related intent-routing data. These scores are 16-option
routing accuracy, not standard 77-way BANKING77 classification. The gains
measure task adaptation within each base; they do not show a Decision-1
advantage over original Gemma after adaptation.

## Matched task adapters

Both bases used identical task rows, option menus, prompt, optimizer, update
budget and seed-matched initial LoRA tensors. Rank-16 q/k/v/o adapters use NF4
base weights, BF16 computation and an FP32 final-softcap option loss. Each arm
selected its checkpoint and seed on validation before scoring test.

The latest four-epoch study's validation-selected packages are
[Decision-1, seed 42 at update 325](https://huggingface.co/Worthify/decision-1-banking77-lora) and
[original Gemma, seed 42 at update 350](https://huggingface.co/Worthify/gemma4-banking77-control-lora).
The public packages pin their base revisions and adapter hashes. Historical 194-update artifacts remain
separately labeled in the [model card](../MODEL_CARD.md).

BANKING77 used 1,540 training rows and 365 validation rows. The original
comparison ran two epochs and 194 updates; the fresh extension ran four epochs
and 388 updates. Both validate at step zero, every 25 updates and epoch ends.
Every test menu contains the answer and 15 distractors from 77 intents. The
77-intent cluster bootstrap uses 10,000 resamples with seed 20260926.

The earlier selected pair gained 1.20 percentage points with Decision-1. In
the longer run, original Gemma scored 93.54% and Decision-1 scored 93.21%, a
−0.32-point difference with interval −0.94 to +0.29. Both extended seed pairs
favored original Gemma on test. Excluding the same 28 rows with known lexical
overlap with full-weight train, validation or test left −0.36 points on
3,052 rows. Full-weight training already included related intent-routing data.

The extension was chosen after viewing the earlier result, which itself
followed CTU-13. The adapters were restarted because optimizer/RNG state was
not saved. The new trajectory is not byte-identical to the old run from update
97 onward. The [extended method and results](FOUNDATION_BANKING77_EXTENDED_LORA_RESULTS.md)
record this limitation, selection steps and paired outcomes. Original Gemma
has higher absolute validation-accuracy area over 0–388; Decision-1 has higher
baseline-adjusted and 194–388 area. Neither establishes faster convergence.
The reports' historical `faster_learning_gate` field refers only to the
baseline-adjusted area.

CTU-13 uses source-host labels and whole capture scenarios, not a representative
sample of current network traffic. Its selected-pair macro-F1 regression is
retained alongside the BANKING77 gain. Different task metrics are not averaged.

The immutable detailed protocols are included in the pinned model evidence:
[BANKING77](https://huggingface.co/Worthify/Decision-1/blob/48acdb76db1d73e0f481b1e2079e7289db0621cc/evidence/banking77-protocol.md)
and [CTU-13](https://huggingface.co/Worthify/Decision-1/blob/48acdb76db1d73e0f481b1e2079e7289db0621cc/evidence/ctu13-protocol.md).
The Decision-1 model and these evidence files are public.

## Full-weight comparison

The preview trained 35,210 rows for two epochs (4,402 updates); validation
selected update 3,200 using 3,200 separate rows. Frozen Gemma, an earlier task
LoRA and the preview used the same BF16 batch-four scorer on one A100 for the
3,200 held-out rows. The family metric is macro-F1 or accuracy as labeled in
the card. Routing macro-F1 includes all 151 offered labels, including 46 with
no gold example. Authored and supplied-menu families are bounded fixtures.
Bootstrap intervals resample source groups and do not quantify variation
across full-weight training seeds. The second full-weight seed is incomplete.

## Recompute and inspect

```bash
python benchmarks/verify_decision_1.py
python benchmarks/verify_banking77_unadapted.py --evidence results/worthify/foundation-banking77-unadapted-transfer-20260927-v1
pip install -e '.[charts]'
python benchmarks/render_decision_1_charts.py
python benchmarks/render_banking77_before_after.py verify
```

The CPU verifier checks evidence hashes, predictions, all task-adapter accuracy
and macro-F1 values, paired correctness, BANKING77 exposure sensitivity and
intent bootstrap, validation curve summaries, extended checkpoint/seed selection,
extended chart polylines, and all 21 full-weight family
headline scores. Full-weight bootstrap intervals, runtime measurements and
underlying source-data lineage are preserved as hashed recorded evidence; this
verifier does not independently reproduce training or those measurements.
The upstream 69-claim verifier is retained separately and is not evidence of
Decision-1 quality. Hashes establish integrity, not measurement correctness.

The original and extended studies have separate evidence directories and raw
aggregates. Their prediction rows are retained unchanged. The extended SVG/PNG
are bound to the public report by a chart manifest; the verifier recomputes all
six displayed SVG curves from the recorded validation values. PNG rasterization
was separately checked with CairoSVG 2.9.0 during source staging. To add another
study, use a new directory and update evidence, method, verifier and claims
together. A late curve crossing alone does not establish faster convergence.
