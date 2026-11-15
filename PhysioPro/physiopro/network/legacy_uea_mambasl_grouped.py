"""Valid-length grouped execution for the pinned MambaSL UEA adapter."""

from __future__ import annotations

import torch

from .legacy_uea_mambasl import MambaSLLegacyUEAAdapter


class GroupedLengthMambaSLAdapter(MambaSLLegacyUEAAdapter):
    """Remove right padding before each call to the unchanged upstream model."""

    execution_mode = "grouped_valid_length"

    def forward(
        self,
        values: torch.Tensor,
        legacy_timestamps: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        del legacy_timestamps
        lengths = valid.sum(dim=1).to(dtype=torch.long)
        if torch.any(lengths < 1):
            raise ValueError("Every legacy UEA sample must retain at least one row.")
        outputs = []
        indices = []
        for length in torch.unique(lengths, sorted=True):
            selected = torch.where(lengths == length)[0]
            length_int = int(length.item())
            compact = values.index_select(0, selected)[:, :length_int, :]
            all_valid = torch.ones(
                compact.shape[:2], device=compact.device, dtype=compact.dtype
            )
            outputs.append(self.model(compact, all_valid))
            indices.append(selected)
        concatenated = torch.cat(outputs, dim=0)
        order = torch.cat(indices, dim=0)
        inverse = torch.argsort(order)
        return concatenated.index_select(0, inverse)
