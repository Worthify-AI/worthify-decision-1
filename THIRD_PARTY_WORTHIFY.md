# Worthify licensing and attribution inventory

Verified September 22, 2026; expanded September 26 for the full-weight and
matched BANKING77 evaluations. Licenses apply separately to code, base
weights, adapters, and datasets; the repository's MIT license does not
relicense external data or model weights.

| Component | Source and license | Preservation and modifications |
|---|---|---|
| OpenJev code | [bonsai/openjev, pinned commit 53e3028](https://github.com/bonsai/openjev/tree/53e3028363509f8533d90fe82d983770da1f6c02), forked from TheoLeeCJ/SemIf; [MIT](https://github.com/bonsai/openjev/blob/53e3028363509f8533d90fe82d983770da1f6c02/LICENSE) | This curated snapshot preserves TheoLeeCJ copyright and LICENSE, selected published evidence, and source-file hashes; it does not distribute the complete research history. Worthify adds Gemma text loading, adapters, training, governed data conversion, evaluation, and publishing. Worthify source additions use MIT. |
| Google Gemma 4 | Immutable base revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; [Apache 2.0](https://ai.google.dev/gemma/apache_2) | Base weights are downloaded separately. Derived adapters use Apache 2.0 with the complete license, attribution, and an explicit fine-tuning notice. Preserve any applicable base NOTICE file using `package --base-notice`; do not invent a NOTICE when none is supplied. |
| CLINC150 | Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* (2019); [source](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | Source license retained by reference in dataset and adapter attribution. Worthify converts utterances and intent labels into runtime candidate sets, reserves unseen labels, groups duplicates, and freezes new splits. These are not original 150-way benchmark results. |
| WANLI | Liu et al., *WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation* (2022); [dataset card](https://huggingface.co/datasets/alisawuffles/WANLI), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Pinned training-source hashes are retained in the model package provenance. Worthify maps entailment/contradiction/neutral to supported/contradicted/insufficient, groups related premises, excludes upstream-test-related components, and freezes new balanced splits. |
| MultiNLI | Williams, Nangia, and Bowman, *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference* (2018); [pinned NYU dataset card](https://huggingface.co/datasets/nyu-mll/multi_nli/blob/da70db2af9d09693783c3320c4249840212ee221/README.md). The card describes mixed source terms: the Open American National Corpus license for most text, and CC BY 3.0, CC BY-SA 3.0, or public-domain status for specified fiction sources. | The full-weight v9 manifest pins the source revision and file hash. Worthify converts selected premise/hypothesis rows to declared evidence choices and retains source-level attribution. Do not label the whole corpus with one CC license. |
| BANKING77 | Casanueva et al. (2020), [PolyAI BANKING77](https://huggingface.co/datasets/PolyAI/banking77); [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). | The matched task-LoRA follow-up pins publisher revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Worthify uses publisher training utterances for the task adapters and converts the official 3,080-row test into deterministic 16-option routing. This is not the original 77-way classification benchmark. |
| Doom demonstration | [ViZDoom 1.3.1](https://github.com/Farama-Foundation/ViZDoom/tree/1.3.1), original code MIT with separately licensed engine components; bundled Freedoom 0.13, BSD 3-Clause | Engine/WAD binaries are installed separately. Recorded frames use the explicitly selected Freedoom assets. The gallery carries the Freedoom BSD-3-Clause license in the original media package. No original commercial Doom WAD is used. |
| Falling-block demonstration | Original Worthify code/drawings, MIT | A reduced placement-based Tetris-style example, without official artwork, music, branding, or copied game code. Pillow is an external rendering dependency. |

The full-weight preview uses attributed CLINC150, WANLI, MultiNLI, and
Worthify-authored records; the separate BANKING77 task LoRAs use publisher
training utterances. Company records and Jev-generated labels are excluded.
Source snapshots and converted prompt text stay outside the Git repository.
Public benchmark evidence may include compact IDs, labels, option
probabilities, and measurements without source text. Dataset attribution
accompanies models and adapters without claiming that the dataset terms are
Apache 2.0.

The project reproduces OpenJev's published runtime-decision methodology. It does not reproduce Jev's undisclosed training or imply endorsement by its creators. [TypeSafe's master customer agreement](https://typesafe.ai/legal/mca) restricts imitation training using the service or its outputs; this project does not obtain training labels from that service. Upstream public comparison artifacts retain their original attribution and evaluation-only role.

The adapter release packager includes only adapters, configuration, model cards, attribution, licenses, checksums, and aggregate evidence. It excludes base weights, source data, row-level predictions, caches, and credentials. The separate full-weight preview export contains derived base weights under its own verification and attribution process.

The small retained upstream baseline evidence also records frozen Qwen3.5-4B
(`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) and Qwen3-Reranker-4B
(`22e683669bc0f0bd69640a1354a6d0aebcfeede5`) measurements, public
[TypeSafe evaluations](https://evals.typesafe.ai/), the public
[Every parallel-judgment lab](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/),
and WANLI. Those aggregate measurements and model-generated token traces are
retained only to preserve the upstream 69-claim verification. They are not
Decision-1 results or training data. The original source and acquisition hashes
remain in the pinned OpenJev revision identified above.
