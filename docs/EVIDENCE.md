# Evidence and method

The model card reports a single full-weight seed and two matched task-LoRA
studies. These files contain prediction IDs, labels and scores; they do not
contain publisher utterances, prompts, source records or model weights.

| Evidence | What it establishes |
|---|---|
| [BANKING77 rows and report](../results/worthify/foundation-banking77-transfer-lora-20260926-v1/) | Four fresh adapters; seeds 42/43 on each base; 3,080 transformed official-test rows. |
| [CTU-13 rows and report](../results/worthify/foundation-transfer-lora-20260926-v1/) | Four fresh adapters on one separate capture scenario, 407 dependent test rows. |
| [Full-weight held-out predictions](../results/worthify/full-weight-seed42-heldout-20260924/) | Frozen base, earlier task LoRA and full-weight preview on the same 3,200 rows. |

## Matched task adapters

Both bases used identical task rows, option menus, prompt, optimizer, update
budget and seed-matched initial LoRA tensors. Rank-16 q/k/v/o adapters use NF4
base weights, BF16 computation and an FP32 final-softcap option loss. Each arm
selected its checkpoint and seed on validation before scoring test.

BANKING77 used 1,540 training rows, 365 validation rows, two epochs and 194
updates. Validation was recorded at step zero, every 25 updates and epoch ends.
Every test menu contains the answer and 15 distractors from 77 intents. The
77-intent cluster bootstrap uses 10,000 resamples with seed 20260926. Of 3,080
rows, 28 have known lexical exposure to the full-weight v9 data. Excluding them
reduces the selected-pair gain from 1.20 to 1.11 percentage points. Full-weight
training already included related intent-routing data. Task choice and this
exposure limit generalization claims.

CTU-13 uses source-host labels and whole capture scenarios, not a representative
sample of current network traffic. Its selected-pair macro-F1 regression is
retained alongside the BANKING77 gain. Different task metrics are not averaged.

The immutable detailed protocols are included in the pinned model evidence:
[BANKING77](https://huggingface.co/Worthify/Decision-1/blob/48acdb76db1d73e0f481b1e2079e7289db0621cc/evidence/banking77-protocol.md)
and [CTU-13](https://huggingface.co/Worthify/Decision-1/blob/48acdb76db1d73e0f481b1e2079e7289db0621cc/evidence/ctu13-protocol.md).
They require reviewer access while the model remains private.

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
pip install -e '.[charts]'
python benchmarks/render_decision_1_charts.py
```

The CPU verifier checks evidence hashes, predictions, all task-adapter accuracy
and macro-F1 values, paired correctness, BANKING77 exposure sensitivity and
intent bootstrap, validation curve summaries, and all 21 full-weight family
headline scores. Full-weight bootstrap intervals, runtime measurements and
underlying source-data lineage are preserved as hashed recorded evidence; this
verifier does not independently reproduce training or those measurements.
The upstream 69-claim verifier is retained separately and is not evidence of
Decision-1 quality. Hashes establish integrity, not measurement correctness.

The longer BANKING77 comparison is pending. To add it, create a new text-free
evidence directory and aggregate, bind their hashes, add a verifier case and
negative tests, and update the card/chart/method together. Preserve the original
194-update result. Do not treat a late curve crossing as faster convergence.
