# Public result summary

## Headline

This project did not find that a larger raw 27B teacher automatically improves 4B medical
question answering. The useful path was staged:

1. train a 4B Medical SFT teacher;
2. let a fresh 4B student generate medical rollouts and receive token-level teacher scores;
3. resume the Medical OPD student with the untouched 4B Base as a general-capability anchor;
4. select an intermediate SAR checkpoint because capability recovery is not monotonic.

## Official full-test comparison

| Checkpoint | MedQA test (3,426) | C-Eval-8 test (1,925) |
|---|---:|---:|
| 4B Base | 71.83% | 85.45% |
| Sequence KD@50 | 71.31% | 82.86% |
| Raw-27B OPD@50 | 72.12% | 84.47% |
| Raw-27B OPD@200 | 68.42% | not completed |

The 50-step raw-27B branch did not establish a robust medical improvement. At 200 steps, the
medical score regressed because more responses failed to finish under the fixed cap. When both
models finished, their paired accuracy was identical.

## 27B domain-adaptation screen

The 27B teacher received 25 Medical SFT steps over 100 Medical-O1 examples, then was evaluated on
a separate MedQA-dev100 set.

| Teacher | Medical accuracy | Valid final answer | Mean output tokens |
|---|---:|---:|---:|
| Raw 27B | 92% | 100% | 881.05 |
| 27B Medical SFT@25 | 81% | 92% | 451.47 |

All eight invalid SFT responses reached the cap without a parseable option. Conditioning on the 92
valid SFT responses reduces the comparison to 84/92 versus 81/92. The supported finding is an
end-to-end answer-termination regression for this small, high-rate recipe, not a proven 11-point
loss of medical knowledge.

## Staged 4B Medical OPD

The 4B Medical SFT trajectory scored 77/63, 82/64 and 78/61 (medical/general) at epochs 1, 2 and 3.
Epoch 2 was selected after observing the screening curve and used as the frozen teacher for a fresh
4B Medical OPD student.

| Checkpoint | MedQA-zh600 | C-Eval300 |
|---|---:|---:|
| 4B Base | 74.50% | 82.00% |
| Medical OPD@50 | 81.17% | 70.33% |
| Medical OPD@100 | 82.50% | 71.67% |
| Medical OPD@200 | 81.50% | 70.00% |
| Medical OPD@300 | 80.33% | 71.33% |

Medical OPD learned a shorter, reliably terminating medical answer policy. At step 300 it produced
zero truncated medical responses, compared with 256/600 for Base. On the subset where both outputs
finished, however, Base remained stronger. The result therefore mixes answer-policy improvement
with domain capability and is not evidence of uniform medical-knowledge transfer.

## Base-anchor SAR

SAR resumed Medical OPD@300, switched the teacher to the untouched 4B Base and trained on general
prompt-only trajectories.

| Checkpoint | MedQA-zh600 | C-Eval300 | Medical truncation |
|---|---:|---:|---:|
| Medical OPD@300 | 80.33% | 71.33% | 0.00% |
| SAR@50 | 83.33% | 74.00% | 5.33% |
| SAR@100 | **85.00%** | 80.33% | 8.33% |
| SAR@200 | 73.83% | 82.67% | 40.50% |
| SAR@250 | 72.67% | **83.00%** | 40.00% |
| SAR@300 | 75.00% | 81.67% | 41.67% |

SAR@100 is the best observed trade-off, not a preregistered endpoint. From step 100 to 300, mean
medical output length rose from 424 to 825 tokens and truncation rose from 8.33% to 41.67%. The
late-stage medical regression is therefore best explained by long Base-like reasoning failing to
close before the 1,024-token medical cap. The longer C-Eval budget still allowed answers to finish.

## Scope and limitations

- The raw-27B table uses complete official MedQA and C-Eval-8 test sets.
- The 600/300 curves are a fixed, row-disjoint protocol diagnostic, not an untouched official test.
- Epoch 2 and SAR@100 were selected after observing their trajectories.
- Lower reverse KL did not imply higher downstream accuracy.
- Aggregate metrics are published; raw benchmark text, predictions and checkpoint URIs are not.

The machine-readable source for the public tables and plots is
[`../results/final_metrics.json`](../results/final_metrics.json).
