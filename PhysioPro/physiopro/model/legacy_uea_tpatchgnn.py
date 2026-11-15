"""Legacy train-es wrapper for the t-PatchGNN UEA adaptation."""

from __future__ import annotations

import torch
from torch import nn

from .legacy_uea import LegacyBaselineMaskTS
from .legacy_uea_hi_patch import merge_legacy_inputs


class LegacyTPatchGNNMaskTS(LegacyBaselineMaskTS):
    def _build_network(
        self,
        network: nn.Module,
        task: str,
        out_ranges=None,
        out_size: int = 1,
        aggregate: bool = True,
        output_size_multiplier: int = 1,
    ) -> None:
        if task != "multiclassification" or out_ranges is not None or not aggregate:
            raise ValueError("t-PatchGNN adaptation supports aggregate multiclassification only.")
        self.network = network
        self.aggregate = True
        self.out_ranges = None
        self.act_out = nn.LogSoftmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.fill_nan_type != "merge":
            raise ValueError("t-PatchGNN adaptation requires fill_nan_type='merge'.")
        values, times, valid = merge_legacy_inputs(self, inputs)
        return self.act_out(self.network(values, times, valid))
