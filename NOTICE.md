# Notice and attribution

MedDistill-OPD is an independent experiment implementation by Wang Hao.

The staged Medical SFT -> Medical OPD -> Base-anchor OPD experiment was informed by the public
`agentic-rl-lab/02-opd` materials by KMnO4-zx. That repository is distributed under the Apache
License 2.0. This project retains the same license and records the reference here rather than
presenting the experimental framing as an unrelated original invention.

The implementation in this repository adds and changes substantial behavior, including:

- a raw 27B-to-4B teacher hypothesis and its negative-result diagnosis;
- bounded 27B Medical SFT screening before downstream distillation;
- exact and near-duplicate filtering against medical evaluation splits;
- fixed official-test and protocol-diagnostic evaluation boundaries;
- explicit token/logprob/mask contracts and finite-value checks;
- paid-run confirmation, usage ledgers, caching and optimizer-aware resume validation;
- checkpoint-curve analysis that separates accuracy from answer-termination effects.

Method background:

- Generalized Knowledge Distillation for Auto-regressive Sequence Models (2023)
- MiniLLM: Knowledge Distillation of Large Language Models (2023)
- Reference experiment: https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/02-opd

Datasets and model weights are not covered by this repository's Apache-2.0 license. Users must
review and comply with the original licenses and terms for Qwen, Medical-O1, MedQA, C-Eval and
Alpaca-ZH.
