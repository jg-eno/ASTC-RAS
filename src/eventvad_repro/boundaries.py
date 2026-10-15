"""Boundary detection and conversion to canonical frame cuts."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BoundaryConfig:
    smooth_seconds: float = 2.0
    mad_multiplier: float = 3.0
    min_gap_seconds: float = 2.0
    mode: str = "paper_reconstruction"


def detect_boundaries(features: np.ndarray, fps: float, config: BoundaryConfig = BoundaryConfig()) -> tuple[list[int], dict]:
    values = np.asarray(features, dtype=np.float32)
    n = len(values)
    diagnostics: dict = {"raw_boundaries": [], "window": 0}
    if n < 2 or fps <= 0:
        return [], diagnostics
    differences = np.diff(values, axis=0)
    squared = np.einsum("ij,ij->i", differences, differences)
    norms = np.linalg.norm(values[:-1], axis=1) * np.linalg.norm(values[1:], axis=1)
    signal = squared + (1 - np.einsum("ij,ij->i", values[:-1], values[1:]) / np.maximum(norms, 1e-12))
    window = max(3, int(round(fps * config.smooth_seconds)))
    diagnostics["window"] = window
    if len(signal) < max(window, 5):
        return [], diagnostics
    try:
        from scipy.signal import savgol_filter
        # SciPy permits even windows; released code uses 60 at 30fps.
        smoothed = savgol_filter(signal, window_length=window, polyorder=2)
    except ValueError:
        return [], diagnostics
    local_mean = np.convolve(smoothed, np.ones(window) / window, mode="valid")
    ratio = smoothed[window - 1:] / np.maximum(local_mean, 1e-6)
    median = float(np.median(ratio)); mad = float(np.median(np.abs(ratio - median)))
    threshold = median + config.mad_multiplier * mad
    candidates = (np.flatnonzero(ratio > threshold) + window // 2 + 1).tolist()
    minimum_gap = int(round(config.min_gap_seconds * fps))
    merged: list[int] = []
    for candidate in candidates:
        if merged and candidate - merged[-1] < minimum_gap:
            merged[-1] = candidate
        else:
            merged.append(candidate)
    diagnostics.update({"signal": signal.tolist(), "smoothed": smoothed.tolist(), "ratio": ratio.tolist(),
                        "threshold": threshold, "raw_boundaries": candidates})
    return [cut for cut in merged if 0 < cut < n], diagnostics
