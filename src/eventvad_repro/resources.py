"""MIG-safe resource admission checks."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


class ResourceBlocked(RuntimeError):
    pass


def preflight(required_gib: float, scratch: str | Path = ".") -> dict:
    scratch_path = Path(scratch)
    existing_scratch = scratch_path
    while not existing_scratch.exists() and existing_scratch.parent != existing_scratch:
        existing_scratch = existing_scratch.parent
    report = {"required_free_gib": required_gib, "cuda_available": False,
              "scratch": str(scratch_path), "scratch_exists": scratch_path.is_dir(),
              "scratch_free_bytes": shutil.disk_usage(existing_scratch).free,
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}
    try:
        import torch
        report.update({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                       "cuda_available": bool(torch.cuda.is_available())})
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            free, total = torch.cuda.mem_get_info(device)
            matrix = torch.ones((32, 32), device=device) @ torch.ones((32, 32), device=device)
            torch.cuda.synchronize(device)
            report.update({"device": torch.cuda.get_device_name(device), "free_bytes": free,
                           "total_bytes": total, "capability": torch.cuda.get_device_capability(device),
                           "matrix_check": float(matrix.sum())})
            if free < required_gib * 1024**3:
                raise ResourceBlocked(f"BLOCKED_RESOURCE free={free / 1024**3:.2f}GiB required={required_gib:.2f}GiB")
    except ImportError:
        report["torch"] = "not installed"
    if shutil.which("nvidia-smi"):
        report["nvidia_smi_l"] = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True).stdout
    return report
