"""Embedding pooling and compact serialisation for storage."""

from __future__ import annotations

import numpy as np
import torch


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def to_bytes(vec: np.ndarray) -> bytes:
    """float16 bytes: 768 dims -> 1.5 KB per comment. Precision is ample for drift statistics."""
    return np.asarray(vec, dtype=np.float16).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)
