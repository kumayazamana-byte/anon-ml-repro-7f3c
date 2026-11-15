"""Forecasting adapter that reuses the archived MHA-to-CM encoder."""

from __future__ import annotations

import torch
from torch import nn

from .hippogp_contiformer import HiPPOGPAttention, HiPPOGPEncoder


class HiPPOGPForecastAdapterFaithful(nn.Module):
    """MHA-to-CM P-to-P encoder followed by a coordinate-query decoder.

    The encoder is the same ``HiPPOGPEncoder`` used by the archived Spiral/UEA
    runs.  The separate decoder is required only because forecasting queries
    need not be token aligned with the observed support.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        d_model: int = 64,
        d_inner: int = 128,
        n_layers: int = 1,
        n_head: int = 4,
        d_v: int = 16,
        hippo_order: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model != n_head * d_v:
            raise ValueError("d_model must equal n_head * d_v.")
        common = dict(
            hippo_order=hippo_order,
            basis_backend="analytic",
            basis_grid_size=128,
            memory_normalization="layernorm",
            local_residual=True,
            local_residual_type="self_attn",
            local_residual_init=0.1,
            uncertainty_gate=False,
            mixture_type="default",
            gp_rope="none",
            jitter=1e-4,
            operator_order="mha_to_cm",
        )
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder = HiPPOGPEncoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_v=d_v,
            dropout=dropout,
            normalize_before=False,
            **common,
        )
        # P->R coordinate query interface; the archived encoder itself remains
        # token-aligned and unchanged.
        self.query_decoder = HiPPOGPAttention(
            d_model=d_model,
            n_head=n_head,
            d_v=d_v,
            dropout=dropout,
            local_residual=False,
            **{k: v for k, v in common.items() if k not in {"operator_order", "local_residual"}},
        )
        self.output_proj = nn.Linear(d_model, output_size or input_size)

    @staticmethod
    def _query_batch(query_t: torch.Tensor, batch: int) -> torch.Tensor:
        if query_t.dim() == 1:
            query_t = query_t.unsqueeze(0).expand(batch, -1)
        if query_t.dim() != 2 or query_t.shape[0] != batch:
            raise ValueError("time_steps_to_predict must have shape [Q] or [B, Q].")
        return query_t

    @staticmethod
    def _invalid_mask(observed_mask: torch.Tensor | None, observed_data: torch.Tensor):
        if observed_mask is None:
            return None, observed_data
        if observed_mask.dim() == 3:
            variable_valid = observed_mask.to(dtype=torch.bool)
            if variable_valid.shape != observed_data.shape:
                raise ValueError("observed_mask does not match observed_data.")
            # Missing variables do not contribute values; a timestep is padded
            # only when every variable is missing.
            clean_data = torch.where(variable_valid, observed_data, torch.zeros_like(observed_data))
            valid = variable_valid.any(dim=-1)
        elif observed_mask.dim() == 2:
            if observed_mask.shape != observed_data.shape[:2]:
                raise ValueError("observed_mask does not match observed_data.")
            clean_data = observed_data
            valid = observed_mask.to(dtype=torch.bool)
        else:
            raise ValueError("observed_mask must have shape [B, T] or [B, T, D].")
        return ~valid, clean_data

    def forward(
        self,
        time_steps_to_predict: torch.Tensor,
        observed_data: torch.Tensor,
        observed_tp: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observed_data.dim() != 3:
            raise ValueError("observed_data must have shape [B, T, D].")
        if observed_tp.dim() == 1:
            observed_tp = observed_tp.unsqueeze(0).expand(observed_data.shape[0], -1)
        if observed_tp.shape != observed_data.shape[:2]:
            raise ValueError("observed_tp must have shape [B, T] or [T].")
        query_t = self._query_batch(time_steps_to_predict, observed_data.shape[0])
        invalid, observed_data = self._invalid_mask(observed_mask, observed_data)
        support_valid = (
            torch.ones(observed_data.shape[:2], dtype=torch.bool, device=observed_data.device)
            if invalid is None else ~invalid
        )
        support_min, support_scale = self.query_decoder._time_bounds(observed_tp, support_valid)
        hidden = self.input_proj(observed_data)
        time_bounds = (support_min, support_scale)
        encoded = self.encoder(hidden, observed_tp, mask=invalid, time_bounds=time_bounds)
        query_hidden = self.query_decoder(
            encoded, observed_tp, mask=invalid, query_t=query_t, time_bounds=time_bounds
        )
        # No observed support carries no information; keep arbitrary queries
        # finite and exactly zero for all-padding rows.
        has_support = support_valid.any(dim=1)
        query_hidden = torch.where(
            has_support[:, None, None], query_hidden, torch.zeros_like(query_hidden)
        )
        output = self.output_proj(query_hidden)
        return torch.where(has_support[:, None, None], output, torch.zeros_like(output))


class HiPPOGPForecastAdapterParallel(HiPPOGPForecastAdapterFaithful):
    """UEA Parallel P-to-P encoder with the same forecasting P-to-R decoder."""

    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        d_model: int = 64,
        d_inner: int = 128,
        n_layers: int = 1,
        n_head: int = 4,
        d_v: int = 16,
        hippo_order: int = 16,
        dropout: float = 0.0,
    ) -> None:
        nn.Module.__init__(self)
        if d_model != n_head * d_v:
            raise ValueError("d_model must equal n_head * d_v.")
        common = dict(
            hippo_order=hippo_order,
            basis_backend="analytic",
            basis_grid_size=128,
            memory_normalization="layernorm",
            local_residual=True,
            local_residual_type="self_attn",
            local_residual_init=0.1,
            uncertainty_gate=False,
            mixture_type="default",
            gp_rope="none",
            jitter=1e-4,
            operator_order="parallel",
        )
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder = HiPPOGPEncoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_v=d_v,
            dropout=dropout,
            normalize_before=False,
            **common,
        )
        self.query_decoder = HiPPOGPAttention(
            d_model=d_model,
            n_head=n_head,
            d_v=d_v,
            dropout=dropout,
            local_residual=False,
            **{k: v for k, v in common.items() if k not in {"operator_order", "local_residual"}},
        )
        self.output_proj = nn.Linear(d_model, output_size or input_size)
