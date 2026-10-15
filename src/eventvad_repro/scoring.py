"""Strict score parsing and frozen VideoLLaMA2 scoring boundary."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import os

PROMPT = """Are there any obvious or potential anomalies in the video?
First describe the visible events and explain the evidence relevant to an anomaly.
Then assign an anomaly score from 0 (normal) to 1 (clearly anomalous).
Base the assessment on the video. End with exactly one line:
Anomaly score: <number>
"""
SCORE_LINE = re.compile(r"^Anomaly score:\s*((?:0(?:\.\d+)?|1(?:\.0+)?))\s*$", re.MULTILINE)
BARE_SCORE = re.compile(r"^(?:0(?:\.\d+)?|1(?:\.0+)?)$")


def parse_anomaly_score(response: str) -> float:
    """Accept the documented final line or an unambiguous bare scalar."""
    value = response.strip()
    if BARE_SCORE.fullmatch(value):
        return float(value)
    matches = list(SCORE_LINE.finditer(value))
    if len(matches) != 1 or matches[0].end() != len(value):
        raise ValueError("response must end with exactly one `Anomaly score: <0..1>` line")
    return float(matches[0].group(1))


def uniformly_sample_indices(start: int, end: int, count: int = 16) -> list[int]:
    if end <= start or count < 1:
        raise ValueError("invalid sampling interval")
    import numpy as np
    return np.linspace(start, end - 1, count, dtype=np.int64).tolist()


@dataclass
class FrozenVideoLLaMA2:
    model_path: Path
    max_new_tokens: int = 256
    quantization: str = "none"

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self._model = self._processor = self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return self._model, self._processor, self._tokenizer
        try:
            import torch
            import torch.nn.functional as F
            from videollama2 import model_init
            from videollama2.utils import disable_torch_init
            from transformers.models.siglip.modeling_siglip import SiglipAttention
        except ImportError as error:
            raise RuntimeError("VideoLLaMA2 score environment is not installed") from error
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        # Transformers 4.40 eager SigLIP attention creates an avoidable large fp32 matrix.
        def sdpa_attention(attention, hidden_states, attention_mask=None, output_attentions=False):
            batch, length, _ = hidden_states.shape
            shape = (batch, length, attention.num_heads, attention.head_dim)
            query = attention.q_proj(hidden_states).view(shape).transpose(1, 2)
            key = attention.k_proj(hidden_states).view(shape).transpose(1, 2)
            value = attention.v_proj(hidden_states).view(shape).transpose(1, 2)
            output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                dropout_p=attention.dropout if attention.training else 0.0, scale=attention.scale)
            return attention.out_proj(output.transpose(1, 2).reshape(batch, length, attention.embed_dim)), None
        SiglipAttention.forward = sdpa_attention
        disable_torch_init()
        if not torch.cuda.is_available():
            raise RuntimeError("VideoLLaMA2 inference requires CUDA")
        if self.quantization not in {"4bit", "none"}:
            raise ValueError("quantization must be '4bit' or 'none'")
        arguments = {"device": "cuda:0"}
        if self.quantization == "4bit":
            arguments["load_4bit"] = True
        self._model, self._processor, self._tokenizer = model_init(str(self.model_path), **arguments)
        self._model.eval()
        return self._model, self._processor, self._tokenizer

    def score_video(self, path: str | Path, prompt: str = PROMPT) -> tuple[float, str]:
        """Score a clip, retrying once when the model omits the required score line."""
        try:
            import torch
            from videollama2 import mm_infer
        except ImportError as error:
            raise RuntimeError("VideoLLaMA2 score environment is not installed") from error
        model, processor, tokenizer = self._load()
        video_input = processor["video"](str(path)).to("cuda:0")
        with torch.inference_mode():
            response = mm_infer(video_input, prompt, model=model, tokenizer=tokenizer,
                do_sample=False, max_new_tokens=self.max_new_tokens, modal="video")
            try:
                return parse_anomaly_score(response), response
            except ValueError as initial_error:
                # Short token budgets can truncate the score line after an otherwise
                # useful description. Ask for a compact, schema-compliant response once.
                retry_prompt = (
                    "Assess the video segment for anomalous or criminal activity using visible evidence. "
                    "Return only one number from 0 to 1: 0 means normal and 1 means clearly anomalous. "
                    "Do not include any words, explanation, or punctuation."
                )
                retry_tokens = max(self.max_new_tokens * 2, 256)
                retry_response = mm_infer(video_input, retry_prompt, model=model, tokenizer=tokenizer,
                    do_sample=False, max_new_tokens=retry_tokens, modal="video")
                try:
                    retry_score = parse_anomaly_score(retry_response)
                    # Normalize a bare scalar so downstream summaries and artifacts
                    # retain the same response shape as a successful first attempt.
                    if BARE_SCORE.fullmatch(retry_response.strip()):
                        retry_response = f"Anomaly score: {retry_response.strip()}"
                    return retry_score, retry_response
                except ValueError as retry_error:
                    initial_excerpt = str(response).strip().replace("\n", " ")[:300]
                    retry_excerpt = str(retry_response).strip().replace("\n", " ")[:300]
                    raise ValueError(
                        f"model omitted a valid anomaly score on initial response and retry "
                        f"(initial: {initial_error}; retry: {retry_error}; "
                        f"initial response: {initial_excerpt!r}; retry response: {retry_excerpt!r})"
                    ) from retry_error
