"""CLIP + RAFT feature extraction for the EventVAD reconstruction."""
from __future__ import annotations

import hashlib
import numpy as np
import os
import sys
from argparse import Namespace
from pathlib import Path


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def orthogonal_projection(in_dim: int, out_dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((max(in_dim, out_dim), max(in_dim, out_dim)))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r) + 1e-32)
    return q[:in_dim, :out_dim].astype(np.float32)


def projection_checksum(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()


def fuse_features(clip: np.ndarray, flow_2d: np.ndarray, projection: np.ndarray,
                  clip_weight: float = .75) -> np.ndarray:
    if clip.ndim != 2 or clip.shape[1] != 512 or flow_2d.shape != (len(clip), 2):
        raise ValueError("expected CLIP [N,512] and flow [N,2]")
    if projection.shape != (2, 128):
        raise ValueError("flow projection must be [2,128]")
    fused = np.concatenate([clip_weight * l2_normalize(clip),
                            (1 - clip_weight) * (flow_2d @ projection)], axis=1)
    return fused.astype(np.float32)


class FeatureExtractor:
    """Frozen OpenAI CLIP ViT-B/16 appearance and official RAFT motion features."""
    def __init__(self, clip_weight: float = .75, projection_seed: int = 42,
                 backend: str = "clip_raft", raft_checkpoint: str | None = None, raft_iters: int = 20,
                 clip_model: str = "openai/clip-vit-base-patch16", clip_batch_size: int = 32):
        if not 0 <= clip_weight <= 1:
            raise ValueError("clip_weight must be in [0, 1]")
        self.clip_weight = float(clip_weight)
        self.flow_projection = orthogonal_projection(2, 128, projection_seed)
        if backend != "clip_raft":
            raise ValueError("only the paper-aligned 'clip_raft' backend is supported")
        self.backend, self.raft_checkpoint, self.raft_iters = backend, raft_checkpoint, int(raft_iters)
        self.clip_model, self.clip_batch_size = clip_model, int(clip_batch_size)
        self._raft = self._padder = self._clip = self._clip_processor = None

    def _load_clip(self):
        if self._clip is not None:
            return self._clip, self._clip_processor
        try:
            import torch
            from transformers import CLIPImageProcessor, CLIPModel
        except ImportError as error:
            raise RuntimeError("Transformers CLIP is required for appearance features") from error
        if not torch.cuda.is_available():
            raise RuntimeError("the CLIP feature backend requires CUDA")
        self._clip_processor = CLIPImageProcessor.from_pretrained(self.clip_model)
        self._clip = CLIPModel.from_pretrained(self.clip_model).to("cuda:0").eval()
        return self._clip, self._clip_processor

    def _load_raft(self):
        """Load Princeton RAFT and its official default Things checkpoint once."""
        if self._raft is not None:
            return self._raft, self._padder
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for the RAFT feature backend") from error
        if not torch.cuda.is_available():
            raise RuntimeError("the RAFT feature backend requires CUDA")
        root = Path(os.environ.get("EVENTVAD_RAFT_ROOT", "/tmp/RAFT"))
        checkpoint = Path(self.raft_checkpoint or root / "models" / "raft-things.pth")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RAFT checkpoint not found: {checkpoint}")
        if str(root / "core") not in sys.path:
            sys.path.insert(0, str(root / "core"))
        try:
            from raft import RAFT
            from utils.utils import InputPadder
        except ImportError as error:
            raise RuntimeError(f"RAFT source not importable from {root}") from error
        args = Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0.0)
        model = RAFT(args)
        state = torch.load(checkpoint, map_location="cpu")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        self._raft, self._padder = model.to("cuda:0").eval(), InputPadder
        return self._raft, self._padder

    def extract(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("cannot extract features from zero frames")
        import torch
        clip_model, clip_processor = self._load_clip()
        raft, padder = self._load_raft()
        appearance = []
        for start in range(0, len(frames), self.clip_batch_size):
            pixel_values = clip_processor(images=frames[start:start + self.clip_batch_size], return_tensors="pt")["pixel_values"].to("cuda:0")
            with torch.inference_mode():
                appearance.append(clip_model.get_image_features(pixel_values=pixel_values).float().cpu().numpy())
        flow = []
        previous_rgb = None
        for frame in frames:
            if previous_rgb is None:
                flow.append(np.zeros(2, dtype=np.float32))
            else:
                image1 = torch.from_numpy(np.ascontiguousarray(previous_rgb)).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0")
                image2 = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0")
                with torch.inference_mode():
                    padded = padder(image1.shape)
                    image1, image2 = padded.pad(image1, image2)
                    _, dense = raft(image1, image2, iters=self.raft_iters, test_mode=True)
                    flow.append(padded.unpad(dense).mean(dim=(0, 2, 3)).cpu().numpy().astype(np.float32))
            previous_rgb = frame
        clip = l2_normalize(np.concatenate(appearance, axis=0).astype(np.float32))
        projected_flow = np.asarray(flow, dtype=np.float32) @ self.flow_projection
        return np.concatenate([self.clip_weight * clip, (1 - self.clip_weight) * projected_flow], axis=1).astype(np.float32)
