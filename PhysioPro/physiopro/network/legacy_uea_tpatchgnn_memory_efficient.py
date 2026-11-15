"""Memory-bounded TTCN adapter for the isolated t-PatchGNN repair suite."""

from __future__ import annotations

from types import MethodType
from typing import Optional
from pathlib import Path

import torch
from torch.nn import functional as F

from .legacy_uea_tpatchgnn import TPatchGNNLegacyUEAAdapter


def chunked_ttcn(
    module,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Evaluate the upstream TTCN equation without materializing its full repeat."""
    if chunk_size < 1:
        raise ValueError("ttcn_chunk_size must be positive.")
    outputs = []
    for start in range(0, inputs.shape[0], chunk_size):
        chunk = inputs[start : start + chunk_size]
        chunk_mask = mask[start : start + chunk_size]
        count, length, _ = chunk_mask.shape
        filters = module.Filter_Generators(chunk)
        filters = filters.masked_fill(~chunk_mask.bool(), -1e8)
        weights = F.softmax(filters, dim=-2)
        weights = weights.view(count, length, module.ttcn_dim, -1)
        outputs.append(torch.einsum("nlf,nlcf->nc", chunk, weights))
    return torch.relu(torch.cat(outputs, dim=0) + module.T_bias)


def _chunked_ttcn_method(module, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return chunked_ttcn(module, inputs, mask, module._legacy_uea_ttcn_chunk_size)


class TPatchGNNMemoryEfficientLegacyUEAAdapter(TPatchGNNLegacyUEAAdapter):
    """Pinned upstream adapter with a numerically equivalent chunked TTCN."""

    def __init__(
        self,
        *args,
        ttcn_chunk_size: int = 256,
        upstream_root: Optional[Path] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, upstream_root=upstream_root, **kwargs)
        if ttcn_chunk_size < 1:
            raise ValueError("ttcn_chunk_size must be positive.")
        self.ttcn_chunk_size = int(ttcn_chunk_size)
        self.model._legacy_uea_ttcn_chunk_size = self.ttcn_chunk_size
        self.model.TTCN = MethodType(_chunked_ttcn_method, self.model)
        self.upstream_provenance = {
            **self.upstream_provenance,
            "numerical_repair": {
                "operation": "chunked_ttcn",
                "chunk_size": self.ttcn_chunk_size,
                "equation_changed": False,
                "effective_batch_size_changed": False,
            },
        }
