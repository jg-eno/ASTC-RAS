"""Staged command-line interface for the EventVAD reconstruction."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import os
from pathlib import Path
from typing import Callable

from . import boundaries, graph
from .evaluation import frame_scores, metrics, paint_labels
from .features import FeatureExtractor
from .manifests import canonical_segments, load_videos, validate_segments, validate_videos, write_segments
from .provenance import initialise_run, read_jsonl, resolve_config, write_json, write_jsonl
from .resources import ResourceBlocked, preflight
from .scoring import FrozenVideoLLaMA2, PROMPT
from .video_io import read_all_frames, video_manifest, write_interval_clip


def _config(args): return resolve_config(args.config)

def cmd_preflight(args):
    config = _config(args); stage = config.get("resources", {}).get("preflight_stage", "segmentation")
    required = float(config.get("resources", {}).get("free_gib", {}).get(stage, 12))
    print(json.dumps(preflight(required, config.get("paths", {}).get("scratch", ".")), indent=2))

def cmd_audit_assets(args):
    config = _config(args); paths = config.get("paths", {})
    required = {"model": paths.get("model"), "raft_checkpoint": paths.get("raft_checkpoint")}
    status = {key: {"path": value, "exists": bool(value and Path(value).exists())} for key, value in required.items()}
    if not all(row["exists"] for row in status.values()):
        raise FileNotFoundError(f"unresolved assets: {status}")
    print(json.dumps(status, indent=2))

def cmd_manifest(args):
    # Input JSONL is deliberately explicit; dataset download/annotation conventions are not guessed.
    rows = load_videos(args.input)
    if args.dataset and any(row["dataset"] != args.dataset for row in rows): raise ValueError("dataset mismatch")
    write_jsonl(args.output, rows); print(f"validated {len(rows)} videos: {args.output}")

def cmd_ingest(args):
    source = Path(args.input)
    candidates = ([source] if source.is_file() else sorted(
        item for item in source.rglob("*") if item.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}))
    if not candidates:
        raise FileNotFoundError(f"no supported videos under {source}")
    rows = [video_manifest(item, args.dataset, args.split) for item in candidates]
    validate_videos(rows, require_files=True); write_jsonl(args.output, rows)
    print(f"ingested {len(rows)} videos: {args.output}")

def cmd_segment(args):
    config = _config(args); run = initialise_run(args.run, config, "segment")
    videos = load_videos(args.manifest or run / "dataset_manifest.jsonl")
    produced = []
    for video in videos:
        feature_path = run / "features" / f"{video['video_id']}.npy"
        if not feature_path.exists(): raise FileNotFoundError(f"missing extracted feature cache: {feature_path}")
        import numpy as np
        features = np.load(feature_path)
        if len(features) != int(video["frame_count"]): raise ValueError(f"feature frame mismatch for {video['video_id']}")
        graph_settings = config.get("graph", {})
        graph_config = graph.GraphConfig(
            block_size=int(graph_settings.get("block_size", 200)),
            clip_weight=float(graph_settings.get("clip_weight", .75)),
            time_decay=float(graph_settings.get("time_decay", .6)),
            seed=int(graph_settings.get("projection_seed", 42)),
            qk_dim=int(graph_settings.get("qk_dim", 64)),
        )
        features = graph.propagate(features, graph.build_blockwise_graph(features, graph_config),
            seed=graph_config.seed, qk_dim=graph_config.qk_dim,
            iterations=int(graph_settings.get("iterations", 1)))
        boundary = config.get("boundary", {})
        cuts, diagnostic = boundaries.detect_boundaries(features, video["fps_num"] / video["fps_den"], boundaries.BoundaryConfig(
            smooth_seconds=float(boundary.get("smooth_seconds", 2.0)), mad_multiplier=float(boundary.get("mad_multiplier", 3.0)),
            min_gap_seconds=float(boundary.get("min_gap_seconds", 2.0)), mode=str(boundary.get("mode", "paper_reconstruction"))))
        np.savez_compressed(run / "boundaries" / f"{video['video_id']}.npz", cuts=np.array(cuts), diagnostic=json.dumps(diagnostic))
        produced.extend(canonical_segments(video, cuts, config["_config_hash"]))
    write_segments(run / "segments.jsonl", produced, videos)

def cmd_extract(args):
    config = _config(args); run = initialise_run(args.run, config, "extract")
    videos = load_videos(args.manifest)
    write_jsonl(run / "dataset_manifest.jsonl", videos)
    feature = config.get("feature", {}); maximum = tuple(feature.get("max_resolution", [1280, 720]))
    extractor = FeatureExtractor(float(feature.get("clip_weight", .75)), int(feature.get("projection_seed", 42)),
        str(feature.get("backend", "clip_raft")), config.get("paths", {}).get("raft_checkpoint"),
        int(feature.get("raft_iterations", 20)), str(feature.get("clip_model", "openai/clip-vit-base-patch16")),
        int(feature.get("clip_batch_size", 32)))
    for video in videos:
        output = run / "features" / f"{video['video_id']}.npy"
        if args.resume and output.exists():
            continue
        frames, _ = read_all_frames(video["source_path"], maximum)
        if len(frames) != int(video["frame_count"]):
            raise ValueError(f"decoded frame count changed for {video['video_id']}")
        import numpy as np
        np.save(output, extractor.extract(frames))
    print(f"extracted {len(videos)} feature arrays in {run / 'features'}")

def _model_path(config):
    configured = Path(str(config.get("paths", {}).get("model", "")))
    if configured.is_dir():
        return configured
    environment = os.environ.get("EVENTVAD_MODEL")
    if environment and Path(environment).is_dir():
        return Path(environment)
    local = Path(__file__).resolve().parents[1] / "models" / "VideoLLaMA2.1-7B-16F"
    if local.is_dir():
        return local
    raise FileNotFoundError(f"VideoLLaMA2 model not found: configured={configured}; set EVENTVAD_MODEL")

def cmd_score(args):
    config = _config(args); run = initialise_run(args.run, config, "score")
    videos = load_videos(run / "dataset_manifest.jsonl")
    segments = read_jsonl(run / "segments.jsonl"); validate_segments(segments, videos)
    rows_by_id = {str(row["video_id"]): row for row in videos}
    scoring = config.get("scoring", {})
    project_root = Path(args.config).resolve().parent.parent
    prompt_path = project_root / str(scoring.get("prompt", ""))
    if not prompt_path.is_file():
        prompt_path = project_root / "src" / str(scoring.get("prompt", ""))
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else PROMPT
    runner = FrozenVideoLLaMA2(_model_path(config), int(scoring.get("max_new_tokens", 256)), str(scoring.get("quantization", "none")))
    predictions = []
    with tempfile.TemporaryDirectory(prefix="eventvad-clips-", dir=run) as temporary:
        for segment in segments:
            identifier = str(segment["segment_id"]); output = run / "raw_responses" / f"{identifier.replace(':', '_')}.txt"
            if args.resume and output.exists():
                payload = json.loads(output.read_text())
                previous = payload.get("prediction", {})
                if previous.get("status") == "success":
                    predictions.append(previous)
                    continue
            video = rows_by_id[str(segment["video_id"])]
            try:
                clip = write_interval_clip(video["source_path"], int(segment["start_frame"]), int(segment["end_frame_exclusive"]),
                    Path(temporary) / f"{identifier.replace(':', '_')}.mp4")
                score, response = runner.score_video(clip, prompt)
                prefix = response.rsplit("Anomaly score:", 1)[0].strip()
                prediction = {"segment_id": identifier, "status": "success", "score": score, "summary": prefix}
            except Exception as error:
                response = f"{type(error).__name__}: {error}"; prediction = {"segment_id": identifier, "status": "error", "score": None}
            output.write_text(json.dumps({"prediction": prediction, "response": response}, indent=2) + "\n", encoding="utf-8")
            predictions.append(prediction)
    write_jsonl(run / "predictions.jsonl", predictions)
    failures = sum(item["status"] != "success" for item in predictions)
    if failures:
        raise RuntimeError(f"{failures} scoring failures; inspect {run / 'raw_responses'}")
    print(f"scored {len(predictions)} segments: {run / 'predictions.jsonl'}")

def cmd_evaluate(args):
    config = _config(args); run = initialise_run(args.run, config, "evaluate")
    videos = load_videos(args.manifest or run / "dataset_manifest.jsonl")
    segments, predictions, annotations = read_jsonl(run / "segments.jsonl"), read_jsonl(run / "predictions.jsonl"), read_jsonl(args.annotations)
    validate_segments(segments, videos)
    scores = frame_scores(videos, segments, predictions)
    annotation_map = {str(row["video_id"]): paint_labels(int(row["frame_count"]), [tuple(item) for item in row.get("intervals", [])]) for row in annotations}
    result = metrics(scores, annotation_map, "average_precision" if args.dataset == "xd_violence" else "roc_auc")
    write_json(run / "coverage.json", {key: len(value) for key, value in scores.items()}); write_json(run / "metrics.json", result)
    print(json.dumps(result, indent=2))

def cmd_report(args):
    config = _config(args); run = Path(args.run); metric_file = run / "metrics.json"
    metric_text = metric_file.read_text() if metric_file.exists() else "No metrics generated."
    (run / "REPRODUCTION_REPORT.md").write_text(f"# EventVAD reconstruction report\n\nResult label: `{config.get('result_label')}`\n\n```json\n{metric_text}\n```\n", encoding="utf-8")

def cmd_run_baseline(args):
    cmd_extract(args)
    cmd_segment(args)
    cmd_score(args)
    if args.annotations:
        cmd_evaluate(args)


COMMANDS: dict[str, Callable] = {"preflight": cmd_preflight, "audit-assets": cmd_audit_assets, "ingest": cmd_ingest,
    "manifest": cmd_manifest, "extract": cmd_extract, "segment": cmd_segment, "score": cmd_score,
    "evaluate": cmd_evaluate, "report": cmd_report, "run-baseline": cmd_run_baseline}

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eventvad_repro.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        item = subparsers.add_parser(name); item.add_argument("--config", required=True)
        if name in {"manifest", "ingest"}:
            item.add_argument("--dataset", required=True); item.add_argument("--input", required=True); item.add_argument("--output", required=True)
            if name == "ingest": item.add_argument("--split", default="test")
        elif name in {"segment", "evaluate"}:
            item.add_argument("--run", required=True); item.add_argument("--manifest"); item.add_argument("--dataset", default="ucf_crime")
            if name == "evaluate": item.add_argument("--annotations", required=True)
        elif name == "extract":
            item.add_argument("--run", required=True); item.add_argument("--manifest", required=True); item.add_argument("--resume", action="store_true")
        elif name == "score":
            item.add_argument("--run", required=True); item.add_argument("--resume", action="store_true")
        elif name in {"report", "run-baseline"}:
            item.add_argument("--run", required=True)
            if name == "run-baseline": item.add_argument("--manifest", required=True); item.add_argument("--annotations"); item.add_argument("--dataset", default="ucf_crime"); item.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try: return COMMANDS[args.command](args) or 0
    except (ValueError, FileNotFoundError, ResourceBlocked, NotImplementedError) as error:
        parser.error(str(error))

if __name__ == "__main__": main()
