"""Frame-complete evaluator with explicit UCF-Crime and XD-Violence adapters."""
from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np


def paint_labels(frame_count: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    labels = np.zeros(frame_count, dtype=np.int8)
    for start, end in intervals:
        if not 0 <= start < end <= frame_count:
            raise ValueError(f"invalid label interval [{start}, {end})")
        labels[start:end] = 1
    return labels


def frame_scores(videos: list[dict[str, Any]], segments: list[dict[str, Any]],
                 predictions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    videos_by_id = {str(row["video_id"]): row for row in videos}
    scores = {key: np.full(int(row["frame_count"]), np.nan, dtype=np.float32)
              for key, row in videos_by_id.items()}
    coverage = {key: np.zeros(int(row["frame_count"]), dtype=np.int8) for key, row in videos_by_id.items()}
    predictions_by_id = {str(row["segment_id"]): row for row in predictions}
    if len(predictions_by_id) != len(predictions):
        raise ValueError("duplicate prediction segment_id")
    for segment in segments:
        identifier = str(segment["segment_id"])
        if identifier not in predictions_by_id:
            raise ValueError(f"missing prediction for {identifier}")
        prediction = predictions_by_id[identifier]
        if prediction.get("status") != "success":
            raise ValueError(f"failed prediction for {identifier}")
        score = float(prediction["score"])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"invalid score for {identifier}")
        video_id, start, end = str(segment["video_id"]), int(segment["start_frame"]), int(segment["end_frame_exclusive"])
        if np.any(coverage[video_id][start:end]):
            raise ValueError(f"overlap in {video_id}")
        scores[video_id][start:end] = score; coverage[video_id][start:end] += 1
    for video_id, count in coverage.items():
        if not np.all(count == 1):
            raise ValueError(f"incomplete coverage for {video_id}")
    return scores


def metrics(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], metric: str) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    if scores.keys() != labels.keys():
        raise ValueError("score/label video coverage differs")
    observed_scores = np.concatenate([scores[key] for key in sorted(scores)])
    observed_labels = np.concatenate([labels[key] for key in sorted(labels)])
    result: dict[str, Any] = {"frames": int(len(observed_labels)), "positive_frames": int(observed_labels.sum()),
                              "negative_frames": int(len(observed_labels) - observed_labels.sum())}
    if len(np.unique(observed_labels)) < 2:
        result["status"] = "undefined_single_class"; return result
    result["status"] = "success"
    result["roc_auc"] = float(roc_auc_score(observed_labels, observed_scores))
    if metric == "average_precision":
        result["average_precision"] = float(average_precision_score(observed_labels, observed_scores))
    return result
