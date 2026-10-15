"""Frame-accurate decoding helpers. OpenCV is an optional stage dependency."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator
import hashlib
import json


def probe_video(path: str | Path) -> dict:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required by manifest/extract stages") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode {path}")
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    if count == 0:
        raise ValueError(f"decoded zero frames from {path}")
    # Reopen to avoid trusting metadata frame count.
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {"frame_count": count, "fps_num": round(fps * 1000), "fps_den": 1000,
            "width": width, "height": height, "container_fps": fps}


def video_manifest(path: str | Path, dataset: str, split: str) -> dict:
    """Build a verified manifest row from a decodable local video."""
    source = Path(path).resolve()
    details = probe_video(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "schema_version": 1, "video_id": hashlib.sha256(str(source).encode()).hexdigest()[:16],
        "dataset": dataset, "split": split, "source_path": str(source), "sha256": digest,
        "frame_count": details["frame_count"], "fps_num": details["fps_num"],
        "fps_den": details["fps_den"], "width": details["width"], "height": details["height"],
        "timebase_policy": "native", "container_fps": details["container_fps"],
    }


def frames_for_interval(path: str | Path, start: int, end: int) -> Iterator:
    if start < 0 or end <= start:
        raise ValueError("interval must be a nonempty [start, end) range")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for frame decoding") from error
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for index in range(start, end):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"decode ended at frame {index}; expected {end}")
        yield frame[:, :, ::-1]  # RGB
    capture.release()


def read_all_frames(path: str | Path, max_resolution: tuple[int, int] | None = None) -> tuple[list, float]:
    """Decode an entire video to RGB frames, optionally downscaling without upscaling."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for feature extraction") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if max_resolution:
            maximum_width, maximum_height = max_resolution
            height, width = frame.shape[:2]
            ratio = min(1.0, maximum_width / width, maximum_height / height)
            if ratio < 1.0:
                frame = cv2.resize(frame, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
        frames.append(frame[:, :, ::-1])
    capture.release()
    if not frames:
        raise ValueError(f"decoded zero frames from {path}")
    return frames, fps


def write_interval_clip(source: str | Path, start: int, end: int, destination: str | Path) -> Path:
    """Materialize one canonical [start, end) interval for VideoLLaMA2's path API."""
    if start < 0 or end <= start:
        raise ValueError("interval must be a nonempty [start, end) range")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for scoring") from error
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot decode {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create temporary clip {destination}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for expected in range(start, end):
        ok, frame = capture.read()
        if not ok:
            writer.release(); capture.release()
            raise ValueError(f"decode ended at frame {expected}; expected {end}")
        writer.write(frame)
    writer.release(); capture.release()
    return destination
