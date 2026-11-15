"""Legacy train-es wrapper for the upstream Hi-Patch classifier."""

from __future__ import annotations

import torch
from torch import nn

from .legacy_uea import LegacyBaselineMaskTS


def merge_legacy_inputs(
    owner: LegacyBaselineMaskTS, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    missing_rows = inputs.isnan().any(dim=-1)
    values = []
    times = []
    for index in range(inputs.shape[0]):
        retained = torch.where(~missing_rows[index])[0]
        values.append(inputs[index, retained, :])
        times.append(retained.to(inputs.device).reshape(-1, 1))
    merged_values, lengths = owner.merge(values)
    merged_times, _ = owner.merge(times)
    merged_times = merged_times.squeeze(-1)
    if owner.norm_time_flg:
        merged_times = merged_times / merged_values.shape[1]
    valid_lengths = torch.tensor(lengths, device=inputs.device, dtype=torch.long)
    token_index = torch.arange(merged_values.shape[1], device=inputs.device).unsqueeze(0)
    valid = token_index < valid_lengths.unsqueeze(1)
    return merged_values, merged_times, valid


class LegacyHiPatchMaskTS(LegacyBaselineMaskTS):
    """Keep LegacyBaselineMaskTS training while using Hi-Patch's own head."""

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
            raise ValueError("Hi-Patch legacy adaptation supports aggregate multiclassification only.")
        self.network = network
        self.aggregate = True
        self.out_ranges = None
        self.act_out = nn.LogSoftmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.fill_nan_type != "merge":
            raise ValueError("Hi-Patch legacy adaptation requires fill_nan_type='merge'.")
        values, times, valid = merge_legacy_inputs(self, inputs)
        return self.act_out(self.network(values, times, valid))
