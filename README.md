# ASTC-RAS EventVAD inference

This repository contains the active EventVAD inference implementation and the ASTC-RAS paper reconstruction. The repository root holds project metadata (`pyproject.toml`, `uv.lock`) and this guide. The runnable source tree and its assets live under `src/`.

The primary profile is `src/configs/paper_reconstruction.yaml`. Its result label is `paper_reconstruction`; reconstruction choices for details absent from the released implementation are documented in `src/REPRODUCTION_NOTES.md`. Do not present this profile as an author-reference reproduction.

## Repository structure

- `src/eventvad_repro/`: ingest, CLIP/RAFT feature extraction, graph propagation, segmentation, VideoLLaMA2 scoring, evaluation, and CLI.
- `src/configs/` and `src/prompts/`: smoke and paper-reconstruction settings and prompts.
- `src/assets/`: bundled smoke clip.
- `src/models/`: local VideoLLaMA2 checkpoint (kept out of Git).
- `src/.venv/`: prepared inference environment (kept out of Git).
- `src/scripts/`: preflight and pipeline launch helpers.
- `src/tests/`: deterministic core checks.

Run commands from the repository root. Outputs and benchmark data should be kept outside the repository.

## Requirements and preflight

The prepared environment is `src/.venv/`. To create a replacement Python 3.11 environment, install a PyTorch/torchvision build matching the host CUDA driver first, then install the pinned inference dependencies:

```bash
python3.11 -m venv src/.venv
src/.venv/bin/python -m pip install --upgrade pip
# Install the appropriate PyTorch/torchvision CUDA build for this host here.
src/.venv/bin/python -m pip install -r src/requirements-inference.txt
```

VideoLLaMA2 and benchmark data may have separate access or license requirements.

The feature stage needs CUDA, the Hugging Face OpenAI CLIP ViT-B/16 weights (fetched/cached on first use), and Princeton RAFT source plus the official `raft-things.pth` checkpoint. By default the RAFT checkout/checkpoint are expected at `/tmp/RAFT`; override with `EVENTVAD_RAFT_ROOT` or `paths.raft_checkpoint`. The local VideoLLaMA2 checkpoint is under `src/models/VideoLLaMA2.1-7B-16F`; set `EVENTVAD_MODEL` or configure `paths.model` to use another copy. Set `HF_HOME` if you want a persistent Hugging Face cache.

From the repository root:

```bash
PYTHONPATH=src src/.venv/bin/python -m eventvad_repro.cli --help
PYTHONPATH=src src/.venv/bin/python -m eventvad_repro.cli preflight --config src/configs/smoke_4bit.yaml
```

The preflight checks available scratch space for the configured stage; it does not download or validate all model assets. To explicitly check model and RAFT paths, use `audit-assets` with a config whose paths point to the installed assets.

## Smoke inference

The bundled smoke clip exercises the same inference stages as the baseline pipeline. It needs the prepared models, working CUDA/PyTorch, and the RAFT installation described above. Run from the repository root:

```bash
PYTHONPATH=src src/.venv/bin/python -m eventvad_repro.cli ingest \
  --config src/configs/smoke_4bit.yaml --dataset smoke \
  --input src/assets/Robbery137_x264.mp4 \
  --output /tmp/eventvad-smoke/videos.jsonl

PYTHONPATH=src src/.venv/bin/python -m eventvad_repro.cli run-baseline \
  --config src/configs/smoke_4bit.yaml \
  --run /tmp/eventvad-smoke/run \
  --manifest /tmp/eventvad-smoke/videos.jsonl
```

The smoke profile uses seed 42, CLIP ViT-B/16 appearance features, RAFT motion features, graph propagation, boundary segmentation, and 4-bit VideoLLaMA2 scoring. Its prompt asks for an activity summary and anomaly score. Each successful row in `predictions.jsonl` contains the segment score and summary. The first CLIP use may download model weights.

## UCF-Crime and XD-Violence inference

Use videos and annotations obtained through their authorized sources; they are not bundled. Create a manifest from a local split (use `xd_violence` for XD-Violence):

```bash
PYTHONPATH=src src/.venv/bin/python -m eventvad_repro.cli ingest \
  --config src/configs/paper_reconstruction.yaml \
  --dataset ucf_crime --split test \
  --input /data/UCF-Crime/test \
  --output /data/manifests/ucf_test.jsonl

PYTHONPATH=src src/.venv/bin/python -m eventvad_repro.cli run-baseline \
  --config src/configs/paper_reconstruction.yaml \
  --run /data/eventvad/runs/ucf-test \
  --manifest /data/manifests/ucf_test.jsonl \
  --annotations /data/manifests/ucf_test_labels.jsonl \
  --dataset ucf_crime
```

For XD-Violence, substitute its video directory and annotation manifest, and set `--dataset xd_violence`. The annotations JSONL is one row per video, for example `{"video_id":"video-001","intervals":[[120,240],[500,620]]}`; intervals are zero-based `[start,end)` frame ranges, and videos must correspond to the manifest IDs. Evaluation reports frame-level ROC-AUC for UCF-Crime and additionally average precision for XD-Violence. The manifest and interval contracts are implemented in `src/eventvad_repro/manifests.py` and `evaluation.py`.

The pipeline uses zero-based source-frame intervals `[start_frame, end_frame_exclusive)`. It extracts features, propagates graph information, creates a gap-free segment list, scores each segment clip, and optionally computes metrics. Runs are rooted outside Git and include resolved configuration/provenance, features, boundaries, predictions, raw responses, coverage, metrics, and a reproduction report. These reconstruction outputs should be interpreted alongside `src/REPRODUCTION_NOTES.md`.
