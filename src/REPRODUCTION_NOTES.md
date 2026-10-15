# Reproduction notes and discrepancy ledger

## Result labels

| Label | Meaning |
|---|---|
| `smoke` | Installation or execution check. Never report it as a baseline metric. |
| `paper_reconstruction` | Follows the paper while declaring choices absent from released code. |
| `author_reference` | Reserved for recovered, attributable author settings. |

## Source ledger

| Item | Evidence | Chosen resolution | Test/status |
|---|---|---|---|
| Upstream source | Public EventVAD repository at pinned revision `25cacd88a82af389776d2b397239f39961ac2d27` | Keep revision identifier and discrepancy notes; omit a duplicate local checkout from the active source tree | Revision recorded in config |
| `graph_propagation` | The released call site imports a function absent from `graph_operations.py`. | Paper Eq. (5)--(8) specifies `paper_reconstruction_v1`: independent QR-orthogonal Q/K/V, scaled neighbor attention, Eq. (4) edge weighting, residual, one update, and global centering. The random seed is not published; config records seed 42. | Implemented and exercised on CPU |
| Graph settings | Release has `clip_weight=.8`, `time_decay=.05`; paper says `.75/.6`. | Primary config uses paper values; released settings require a diagnostic config. | Config review |
| RAFT | Upstream makes RAFT with `/path/raft-things.pth` and does not load it. | Runnable reconstruction uses the official Princeton RAFT source and default `raft-things.pth` checkpoint, strict-loaded. Appearance uses frozen OpenAI CLIP ViT-B/16. | Local extraction implemented; CLIP weights are downloaded at first use |
| Prompt | Upstream scorer uses the literal placeholder `prompt`; full wording is not attributable. | Configured prompt requests concise event description plus anomaly score; smoke uses its own shorter prompt. | Strict final-score parser |
| Scoring | Local scorer defaults to 4-bit, 32 tokens, path-keyed output. | Scores materialized canonical interval clips using unique `segment_id`; smoke uses a bare-scalar score contract, primary uses the paper-reconstruction prompt. | Local adapter implemented |
| Intervals | Upstream writer is inclusive while evaluator slices exclusive. | Canonical zero-based `[start,end)` throughout package. | Coverage unit test |
| Graph ordering | Upstream boundary code iterates NetworkX insertion order. | Arrays are always chronological frame order. | CPU graph test |

## Execution record

Record each actual run below before publishing a metric.

| Date | Dataset/split | Result label | Config hash | Model/asset hashes | GPU/resource profile | Status |
|---|---|---|---|---|---|---|
| | | | | | | Not run |

## Current blockers

1. Obtain official RGB test videos, split lists, and frame annotation adapters; no benchmark data is included here.
2. For an author-reference claim, recover the author graph propagation, strict pretrained RAFT load, and original scoring prompt/settings. The runnable local adapters are explicitly a reconstruction.

The code can validate manifests, graph behavior, segmentation from cached features, score parsing, coverage, and metrics without those assets. No benchmark result has been generated.
