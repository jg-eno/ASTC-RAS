"""Dataset-independent frame manifest and canonical segment contracts."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .provenance import read_jsonl, write_jsonl

VIDEO_REQUIRED = {"schema_version", "video_id", "dataset", "split", "source_path", "sha256",
                  "frame_count", "fps_num", "fps_den", "width", "height", "timebase_policy"}
SEGMENT_REQUIRED = {"video_id", "segment_id", "source_sha256", "start_frame", "end_frame_exclusive",
                    "segmentation_config_hash", "status"}


def validate_videos(rows: list[dict[str, Any]], require_files: bool = False) -> None:
    identifiers: set[str] = set()
    for row in rows:
        missing = VIDEO_REQUIRED - row.keys()
        if missing:
            raise ValueError(f"video manifest missing {sorted(missing)}")
        identifier = str(row["video_id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate video_id: {identifier}")
        identifiers.add(identifier)
        if int(row["frame_count"]) < 1 or int(row["fps_num"]) < 1 or int(row["fps_den"]) < 1:
            raise ValueError(f"invalid frame/fps fields for {identifier}")
        if require_files and not Path(row["source_path"]).is_file():
            raise FileNotFoundError(row["source_path"])


def validate_segments(rows: list[dict[str, Any]], videos: list[dict[str, Any]]) -> None:
    expected = {str(item["video_id"]): int(item["frame_count"]) for item in videos}
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in expected}
    identifiers: set[str] = set()
    for row in rows:
        missing = SEGMENT_REQUIRED - row.keys()
        if missing:
            raise ValueError(f"segment manifest missing {sorted(missing)}")
        key, segment = str(row["video_id"]), str(row["segment_id"])
        if key not in grouped:
            raise ValueError(f"unknown segment video_id: {key}")
        if segment in identifiers:
            raise ValueError(f"duplicate segment_id: {segment}")
        identifiers.add(segment)
        start, end = int(row["start_frame"]), int(row["end_frame_exclusive"])
        if not 0 <= start < end <= expected[key]:
            raise ValueError(f"invalid interval [{start}, {end}) for {segment}")
        grouped[key].append(row)
    for video_id, items in grouped.items():
        position = 0
        for item in sorted(items, key=lambda row: int(row["start_frame"])):
            if int(item["start_frame"]) != position:
                raise ValueError(f"segments for {video_id} are not gap-free at frame {position}")
            position = int(item["end_frame_exclusive"])
        if position != expected[video_id]:
            raise ValueError(f"segments for {video_id} finish at {position}, expected {expected[video_id]}")


def canonical_segments(video: dict[str, Any], boundaries: list[int], config_hash: str) -> list[dict[str, Any]]:
    count = int(video["frame_count"])
    cuts = sorted({int(value) for value in boundaries if 0 < int(value) < count})
    points = [0, *cuts, count]
    return [{"video_id": video["video_id"], "segment_id": f"{video['video_id']}:{index:05d}",
             "source_sha256": video["sha256"], "start_frame": start, "end_frame_exclusive": end,
             "segmentation_config_hash": config_hash, "status": "pending"}
            for index, (start, end) in enumerate(zip(points, points[1:]))]


def load_videos(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    validate_videos(rows)
    return rows


def write_segments(path: str | Path, rows: list[dict[str, Any]], videos: list[dict[str, Any]]) -> None:
    validate_segments(rows, videos)
    write_jsonl(path, rows)
