"""Adaptive temporal logit pooling for the isolated legacy UEA suite."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .legacy_uea import LegacyBaselineMaskTS


class AdaptiveLogitLegacyMaskTS(LegacyBaselineMaskTS):
    """Use a padding-aware MambaSL-style temporal logit readout."""

    def __init__(
        self,
        *args: Any,
        adaptive_pooling_heads: int = 8,
        pooling_mode: str = "adaptive_logit",
        **kwargs: Any,
    ) -> None:
        if pooling_mode != "adaptive_logit":
            raise ValueError("AdaptiveLogitLegacyMaskTS requires pooling_mode='adaptive_logit'.")
        if adaptive_pooling_heads < 1:
            raise ValueError("adaptive_pooling_heads must be positive.")
        self.pooling_mode = pooling_mode
        self.adaptive_pooling_heads = int(adaptive_pooling_heads)
        self._pooling_diagnostic_sums: dict[str, float] = {}
        self._pooling_diagnostic_count = 0
        # MaskTS validates its historical modes. Passing ``last`` preserves that
        # code while this subclass installs the isolated adaptive path.
        super().__init__(*args, pooling_mode="last", **kwargs)
        self.pooling_mode = pooling_mode
        self.reset_pooling_diagnostics()

    def _build_network(
        self,
        network: nn.Module,
        task: str,
        out_ranges=None,
        out_size: int = 1,
        aggregate: bool = True,
        output_size_multiplier: int = 1,
    ) -> None:
        super()._build_network(
            network,
            task=task,
            out_ranges=out_ranges,
            out_size=out_size,
            aggregate=aggregate,
            output_size_multiplier=output_size_multiplier,
        )
        self.adaptive_gate = nn.Linear(
            network.output_size, self.adaptive_pooling_heads, bias=True
        )
        nn.init.zeros_(self.adaptive_gate.weight)
        nn.init.ones_(self.adaptive_gate.bias)

    def reset_pooling_diagnostics(self) -> None:
        self._pooling_diagnostic_sums = {
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "max_weight": 0.0,
            "effective_support": 0.0,
        }
        self._pooling_diagnostic_count = 0

    def pooling_diagnostics(self) -> dict[str, float | int]:
        count = self._pooling_diagnostic_count
        if count == 0:
            return {"sample_count": 0}
        return {
            "sample_count": count,
            **{
                key: value / count
                for key, value in self._pooling_diagnostic_sums.items()
            },
        }

    def _adaptive_logits(
        self, seq_out: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_scores = self.adaptive_gate(seq_out).amax(dim=-1)
        gate_scores = gate_scores.masked_fill(~valid, -torch.inf)
        weights = torch.softmax(gate_scores, dim=1)
        token_logits = self.fc_out(seq_out)
        logits = (token_logits * weights.unsqueeze(-1)).sum(dim=1)
        return logits, weights

    def _record_pooling_diagnostics(
        self, weights: torch.Tensor, valid_lengths: torch.Tensor
    ) -> None:
        if self.training:
            return
        safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
        entropy = -(weights * safe.log()).sum(dim=1)
        normalizer = valid_lengths.clamp_min(2).to(weights.dtype).log()
        normalized = torch.where(valid_lengths > 1, entropy / normalizer, torch.ones_like(entropy))
        values = {
            "entropy": entropy,
            "normalized_entropy": normalized,
            "max_weight": weights.amax(dim=1),
            "effective_support": entropy.exp(),
        }
        for key, value in values.items():
            self._pooling_diagnostic_sums[key] += float(value.detach().sum().cpu())
        self._pooling_diagnostic_count += int(weights.shape[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.fill_nan_type != "merge":
            raise ValueError("Adaptive legacy pooling requires fill_nan_type='merge'.")
        missing_rows = inputs.isnan().any(dim=-1)
        values = []
        times = []
        for index in range(inputs.shape[0]):
            retained = torch.where(~missing_rows[index])[0]
            values.append(inputs[index, retained, :])
            times.append(retained.to(inputs.device).reshape(-1, 1))
        merged_values, lengths = self.merge(values)
        merged_times, _ = self.merge(times)
        merged_times = merged_times.squeeze(-1)
        if self.norm_time_flg:
            merged_times = merged_times / merged_values.shape[1]

        valid_lengths = torch.tensor(lengths, device=inputs.device, dtype=torch.long)
        positions = torch.arange(merged_values.shape[1], device=inputs.device).unsqueeze(0)
        valid = positions < valid_lengths.unsqueeze(1)
        attn_mask = ~(valid.unsqueeze(1) & valid.unsqueeze(2))
        seq_out, _ = self.network(merged_values, merged_times, mask=attn_mask)
        logits, weights = self._adaptive_logits(seq_out, valid)
        self._record_pooling_diagnostics(weights, valid_lengths)
        return self.act_out(logits.squeeze(-1))
