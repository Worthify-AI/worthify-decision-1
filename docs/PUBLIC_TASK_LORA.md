# Public task-LoRA CLI

`worthify-lora` trains and scores a user-supplied decision task with the native
next-token option readout. It does not fetch data or publish an adapter. Training
needs one visible CUDA GPU and a Hugging Face model ID plus its 40-character
commit revision. Local model bundles are not supported by this public CLI.

In your activated Python environment, install from the repository root:

```bash
pip install -e '.[train]'
```

Each train and validation file is JSONL. Every row has a nonempty `id`, `state`
(string, object, or array), `question`, 2–16 options, and `gold_option_id`.
Options are objects with `id` and `description`. The gold ID must name an option.
Optional `group_id` values let the CLI reject a source group shared across train
and validation. It also rejects repeated IDs and normalized state text across
those splits. Keep test rows separate from model selection.

```json
{"id":"case-1","state":"The payment failed twice.","question":"Which route applies?","options":[{"id":"billing","description":"Billing support"},{"id":"general","description":"General support"}],"gold_option_id":"billing"}
```

The example uses Worthify's seed-42 preview at an immutable revision. Hugging
Face access depends on that model repository's permissions.

```bash
MODEL_ID=Worthify/Decision-1
MODEL_REVISION=cab55818cfb6f4163f2b2597bed31cceb6ba2030

CUDA_VISIBLE_DEVICES=0 worthify-lora train \
  --model "$MODEL_ID" --revision "$MODEL_REVISION" \
  --train my-train.jsonl --validation my-validation.jsonl \
  --seed 42 --epochs 2 --output runs/new-task-run

CUDA_VISIBLE_DEVICES=0 worthify-lora verify \
  --model "$MODEL_ID" --revision "$MODEL_REVISION" \
  --run runs/new-task-run --validation my-validation.jsonl \
  --output runs/new-verification.json

CUDA_VISIBLE_DEVICES=0 worthify-lora evaluate \
  --model "$MODEL_ID" --revision "$MODEL_REVISION" \
  --run runs/new-task-run --validation my-validation.jsonl \
  --input my-test.jsonl --output runs/new-test-scores.jsonl
```

`score` accepts the same model and input arguments. It can score an adapter with
`--run` and `--validation`, or a pinned frozen base without those arguments.
`evaluate` requires gold IDs and reports accuracy. Every output path is
create-only. An adapter run records the base identity, hashes, seed, epochs,
validation history, selected epoch, initial adapter tensor hash, and a fixed
logit reference. Matching seeds reset adapter initialization after base loading;
compare the initial hashes when pairing compatible base architectures. Verify and
adapter scoring freshly load the adapter and compare those logits. The emitted
option probabilities are conditional scores, not calibrated confidence.

Saved PEFT configs pin the base commit. Use `worthify-lora`
to load Gemma adapters: the saved `numeric-policy.json` requires an FP32 final
softcap hook that ordinary PEFT loading alone does not install.

See [the evidence guide](EVIDENCE.md) for the measured comparisons and [the model card](../MODEL_CARD.md) for model limitations.
