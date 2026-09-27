# Predeclared BANKING77 extended LoRA comparison

This is a fresh, matched extension of the exploratory comparison specified in
`FOUNDATION_BANKING77_TRANSFER_EVAL.md`, whose SHA-256 is
`10ed4ce120def3901b2256d2b90df75e86de7f1855f53f403b511fda86b7f14c`.
The earlier 194-update adapters store no AdamW optimizer or RNG state. They
cannot be resumed as an equivalent continuation. All four adapters here start
from fresh, seed-matched LoRA initialization.

Freeze this document and trainer hash before launching any run. Reuse exactly
the previous private BANKING77 low-data manifest (SHA-256
`b9e434686de8ad0670bb3ca5223df78cee39af9697d09e1b068ed3ae114f714b`),
train and validation rows, sealed official test, source/model revisions, prompt,
16-option conversion, base quantization, numeric policy, seed pair 42/43,
LoRA configuration, randomized order, global/microbatch size, AdamW learning
rate and 50-step warmup, gradient clipping, and maximum 2,048 tokens. Expose
exactly one CUDA GPU per scorer process. Use create-only private output paths.

The sole training change is **four complete epochs**, 388 optimizer updates
for the frozen 1,540-row training split. Validate at step zero, every 25
updates, every epoch end (97, 194, 291, 388), and the last step, deduplicated.
The selected checkpoint must follow at least one full unique training pass.
Within each run choose validation accuracy, then macro-F1 across the 77
original intents, then lower cross-entropy, then earliest update. Choose a
seed per base using the same rule, with lower seed on an exact tie. Lock all
four runs and fresh-process validation parity before opening the sealed test.

The primary endpoint for this extension is the tuned-minus-original official
test accuracy of the two validation-selected adapters under this fixed
four-epoch budget. Report both selected scores, paired correctness, and the
77-intent bootstrap interval using the original 10,000-resample method.
Secondary endpoints: both seed-pair test differences, 77-intent macro-F1,
per-intent accuracy, fixed step-388 validation accuracy, validation accuracy
AUC and baseline-adjusted AUC from zero through 388, and the later-phase AUC
over steps 194 through 388. Report the earliest validation-grid step at which
each run reaches 80% accuracy and wall time. Compare the first 194 steps to
the original experiment only as a reproducibility check. Apply the original
exposure-excluded sensitivity analysis to the same 28 flagged test rows.

The exploratory task selection after a negative CTU-13 result, 16-option
conversion, known v9 exposure, lexical decontamination limits, and single
full-weight base seed remain limitations. A crossing at a selected checkpoint
does not by itself show a faster learning rate; assess baseline-adjusted AUC
and the later-phase curve separately. No public claim or release follows
from this protocol alone.
