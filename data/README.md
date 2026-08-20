# Data boundary

Raw questions, completions, predictions and retrieval passages are intentionally excluded from
the public repository. Run `medical-opd prepare-data` to build local artifacts.

| Dataset | Revision used | Purpose | License note |
|---|---|---|---|
| `FreedomIntelligence/medical-o1-reasoning-SFT` (`zh`) | `fc2c9e8a37b38f38da6d449564a8c350b244aef4` | Medical SFT and prompt-only OPD | Apache-2.0 in the dataset card |
| `bigbio/med_qa` | `ddef95d268cdad413693d634279a9a679d468469` | Medical evaluation | Dataset card reports unknown; review source terms |
| `ceval/ceval-exam` | `85ae5586dbc20cac29d0bef4df66c6569b04c1a0` | General evaluation and label-free SAR prompts | CC BY-NC-SA 4.0 |
| `shibing624/alpaca-zh` | `f39db019a94f8dbea48ab30d2bdc090703284559` | Mixed SFT and historical replay | Treat as CC BY-NC 4.0 / research-only |

The medical corpus is normalized, de-duplicated internally, and filtered for exact and near
overlap against both MedQA dev and test. The public metrics snapshot contains aggregate values
only and does not include benchmark text.
