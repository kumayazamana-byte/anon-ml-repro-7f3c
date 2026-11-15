"""Legacy-clock UEA baselines.

These networks deliberately consume the tensors produced by ``MaskTS.merge``.
They are isolated from the historical ContiFormer and HiPPO-GP implementations.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .tsrnn import NETWORKS
from .contiformer import PositionwiseFeedForward


def valid_tokens(mask: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Return [batch, time] validity from the legacy square attention mask."""
    if mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if mask.ndim == 3:
        return ~mask.all(dim=-1)
    if mask.ndim == 2:
        return ~mask
    raise ValueError("mask must have shape [B, T] or [B, T, T].")


class ContinuousTimeEncoding(nn.Module):
    """Sinusoidal encoding evaluated at legacy ``MaskTS`` timestamps."""

    def __init__(self, d_model: int):
        super().__init__()
        inv_freq = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.d_model = d_model

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        angles = times.unsqueeze(-1).to(self.inv_freq.dtype) * self.inv_freq
        out = torch.zeros(*times.shape, self.d_model, device=times.device, dtype=times.dtype)
        out[..., 0::2] = torch.sin(angles).to(out.dtype)
        out[..., 1::2] = torch.cos(angles).to(out.dtype)
        return out


class ValidBatchNorm1d(nn.Module):
    """BatchNorm over valid tokens only, so right padding cannot change a run."""

    def __init__(self, d_model: int, momentum: float = 0.95):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model, momentum=momentum)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        if valid.any():
            out[valid] = self.norm(x[valid])
        return out


class TSTLegacyLayer(nn.Module):
    """The mvts-transformer BatchNorm encoder block in batch-first form."""

    def __init__(self, d_model: int, n_head: int, d_inner: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_inner)
        self.linear2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = ValidBatchNorm1d(d_model)
        self.norm2 = ValidBatchNorm1d(d_model)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        attn, _ = self.self_attn(x, x, x, key_padding_mask=~valid, need_weights=False)
        x = self.norm1(x + self.dropout1(attn), valid)
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff), valid)
        return torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))


@NETWORKS.register_module("tst_legacy_t")
class TSTLegacyT(nn.Module):
    """TST dagger baseline: a timestamp-aware legacy-clock adaptation, not official TST."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        d_model: int = 64,
        n_head: int = 8,
        n_layers: int = 3,
        d_inner: int = 256,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required.")
        self.project = nn.Linear(input_size, d_model)
        self.time_encoding = ContinuousTimeEncoding(d_model)
        self.layers = nn.ModuleList(
            [TSTLegacyLayer(d_model, n_head, d_inner, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.__output_size = d_model
        self.__hidden_size = d_model

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        valid = valid_tokens(mask, x)
        out = self.project(x) * math.sqrt(self.__output_size)
        out = self.dropout(out + self.time_encoding(t))
        out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
        for layer in self.layers:
            out = layer(out, valid)
        return out, out[:, -1, :]

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


class MHAOnlyEncoderLayer(nn.Module):
    """Exact local-MHA arm of the legacy V3 block after GP memory removal."""

    def __init__(self, d_model: int, d_inner: int, n_head: int, dropout: float, residual_init: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.local_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.local_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=False
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        residual = x
        local, _ = self.local_attn(x, x, x, key_padding_mask=~valid, need_weights=False)
        out = self.layer_norm(residual + self.dropout(self.local_scale * local))
        out = self.pos_ffn(out)
        return torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))


@NETWORKS.register_module("mha_only_v3_ablation")
class MHAOnlyV3Ablation(nn.Module):
    """Timestamp-agnostic strict V3 local-attention ablation."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        d_model: int = 32,
        d_inner: int = 128,
        n_layers: int = 1,
        n_head: int = 4,
        dropout: float = 0.1,
        local_residual_init: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required.")
        self.linear = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList(
            [
                MHAOnlyEncoderLayer(
                    d_model, d_inner, n_head, dropout, local_residual_init
                )
                for _ in range(n_layers)
            ]
        )
        self.__output_size = d_model
        self.__hidden_size = d_model

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        del t  # The original V3 local MHA arm does not consume timestamps.
        valid = valid_tokens(mask, x)
        out = self.linear(x)
        out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
        for layer in self.layers:
            out = layer(out, valid)
        return out, out[:, -1, :]

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


def s5_zoh_discretize(
    lam: torch.Tensor, b_tilde: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official S5 zero-order-hold discretization, supporting batched delta."""
    if delta.ndim == 1:
        delta = delta.unsqueeze(-1)
    if delta.ndim != 2:
        raise ValueError("delta must have shape [batch] or [batch, state].")
    lam_bar = torch.exp(delta * lam.unsqueeze(0))
    b_bar = ((lam_bar - 1.0) / lam.unsqueeze(0)).unsqueeze(-1) * b_tilde.unsqueeze(0)
    return lam_bar, b_bar


def s5_zoh_recurrence_numpy(
    x: np.ndarray,
    delta: np.ndarray,
    valid: np.ndarray,
    lam: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    state_step: np.ndarray,
    reverse: bool = False,
) -> np.ndarray:
    """Direct NumPy reference for the complex S5 ZOH state recurrence.

    This is intentionally a literal loop rather than a vectorized alternative:
    it is used only by the numerical-equivalence test for this PyTorch port.
    """
    batch, length, d_model = x.shape
    state = np.zeros((batch, lam.size), dtype=np.complex128)
    out = np.zeros((batch, length, d_model), dtype=np.float64)
    indices = range(length - 1, -1, -1) if reverse else range(length)
    for index in indices:
        step = delta[:, index:index + 1] * state_step[None, :]
        lam_bar = np.exp(step * lam[None, :])
        b_bar = ((lam_bar - 1.0) / lam[None, :])[:, :, None] * b[None, :, :]
        proposed = lam_bar * state + np.einsum("bph,bh->bp", b_bar, x[:, index, :])
        state = np.where(valid[:, index:index + 1], proposed, state)
        value = 2.0 * np.einsum("hp,bp->bh", c, state).real
        out[:, index, :] = np.where(valid[:, index:index + 1], value, 0.0)
    return out


def _make_dplr_hippo(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch translation of S5's ``make_DPLR_HiPPO`` initialisation."""
    idx = torch.arange(n, dtype=torch.float32, device=device)
    p = torch.sqrt(1.0 + 2.0 * idx)
    # Official S5: A = -tril(P P^T) + diag(arange(N)); then S = A + q q^T.
    # The sign is material because it establishes the left-half-plane S5
    # initialization before the ZOH recurrence.
    a = -torch.tril(p[:, None] * p[None, :]) + torch.diag(idx)
    normal_rank = torch.sqrt(idx + 0.5)
    s = a + normal_rank[:, None] * normal_rank[None, :]
    lam_real = torch.diagonal(s).mean().repeat(n)
    lam_imag, v = torch.linalg.eigh(s.to(torch.complex64) * (-1j))
    return lam_real.to(torch.complex64) + 1j * lam_imag, v, v.conj().transpose(0, 1)


class S5Layer(nn.Module):
    """PyTorch S5 layer using legacy per-token timestamp differences."""

    def __init__(self, d_model: int, state_size: int, blocks: int, dropout: float):
        super().__init__()
        if state_size % blocks:
            raise ValueError("state_size must be divisible by blocks.")
        device = torch.device("cpu")
        block = state_size // blocks
        lam, v, vinv = _make_dplr_hippo(block, device)
        if block % 2:
            raise ValueError("S5 conjugate-symmetric block size must be even.")
        half = block // 2
        lam = lam[:half].repeat(blocks)
        v = torch.block_diag(*([v[:, :half]] * blocks))
        vinv = torch.block_diag(*([vinv[:half, :]] * blocks))
        # With conjugate symmetry the stored complex state has P / 2 entries,
        # while the official B/C parameters are first sampled in the real P
        # dimensional HiPPO basis and then projected through V^{-1}/V.
        self.state_size = state_size // 2
        self.register_buffer("lambda_init", lam, persistent=False)
        self.register_buffer("v", v, persistent=False)
        self.register_buffer("vinv", vinv, persistent=False)
        self.lambda_re = nn.Parameter(lam.real.clone())
        self.lambda_im = nn.Parameter(lam.imag.clone())
        self.b_re = nn.Parameter(torch.empty(state_size, d_model))
        self.b_im = nn.Parameter(torch.empty(state_size, d_model))
        self.c_forward_re = nn.Parameter(torch.empty(d_model, state_size))
        self.c_forward_im = nn.Parameter(torch.empty(d_model, state_size))
        self.c_reverse_re = nn.Parameter(torch.empty(d_model, state_size))
        self.c_reverse_im = nn.Parameter(torch.empty(d_model, state_size))
        self.d = nn.Parameter(torch.empty(d_model))
        self.log_step = nn.Parameter(torch.empty(self.state_size))
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Flax BatchNorm momentum is the old-running-statistic coefficient;
        # PyTorch's argument is the incoming-batch coefficient.
        self.norm = ValidBatchNorm1d(d_model, momentum=0.05)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Flax lecun_normal followed by V^{-1}/V, as in the official implementation.
        nn.init.kaiming_normal_(self.b_re, nonlinearity="linear")
        nn.init.zeros_(self.b_im)
        nn.init.kaiming_normal_(self.c_forward_re, nonlinearity="linear")
        nn.init.kaiming_normal_(self.c_forward_im, nonlinearity="linear")
        nn.init.kaiming_normal_(self.c_reverse_re, nonlinearity="linear")
        nn.init.kaiming_normal_(self.c_reverse_im, nonlinearity="linear")
        nn.init.normal_(self.d)
        nn.init.uniform_(self.log_step, math.log(0.001), math.log(0.1))
        nn.init.kaiming_normal_(self.gate.weight, nonlinearity="linear")
        nn.init.zeros_(self.gate.bias)

    def _params(self, dtype: torch.dtype, device: torch.device):
        lam = torch.complex(self.lambda_re, self.lambda_im).to(device)
        b_raw = torch.complex(self.b_re, self.b_im).to(device)
        b = self.vinv.to(device=device, dtype=b_raw.dtype) @ b_raw
        c_f_raw = torch.complex(self.c_forward_re, self.c_forward_im).to(device)
        c_r_raw = torch.complex(self.c_reverse_re, self.c_reverse_im).to(device)
        c_f = c_f_raw @ self.v.to(device=device, dtype=c_f_raw.dtype)
        c_r = c_r_raw @ self.v.to(device=device, dtype=c_r_raw.dtype)
        step = torch.exp(self.log_step).to(device=device, dtype=lam.real.dtype)
        return lam, b, c_f, c_r, step

    @staticmethod
    def _run_direction(
        x: torch.Tensor,
        delta: torch.Tensor,
        valid: torch.Tensor,
        lam: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        state_step: torch.Tensor,
        reverse: bool,
    ) -> torch.Tensor:
        batch, length, d_model = x.shape
        state = torch.zeros(batch, lam.numel(), dtype=lam.dtype, device=x.device)
        out = torch.zeros(batch, length, d_model, dtype=x.dtype, device=x.device)
        indices = range(length - 1, -1, -1) if reverse else range(length)
        for i in indices:
            # Official S5 learns one positive discretization scale per state.
            # Legacy adjacent timestamp gaps supply the data-dependent factor.
            step = delta[:, i].unsqueeze(-1) * state_step.unsqueeze(0)
            lam_bar, b_bar = s5_zoh_discretize(lam, b, step)
            state_new = lam_bar * state + torch.einsum(
                "bph,bh->bp", b_bar, x[:, i].to(dtype=b_bar.dtype)
            )
            state = torch.where(valid[:, i].unsqueeze(-1), state_new, state)
            y = 2.0 * torch.einsum("hp,bp->bh", c, state).real
            out[:, i] = torch.where(valid[:, i].unsqueeze(-1), y.to(x.dtype), torch.zeros_like(y, dtype=x.dtype))
        return out

    def forward(self, x: torch.Tensor, delta: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        residual = x
        # The public Speech Commands invocation leaves ``prenorm`` at its
        # runner default (true) and enables BatchNorm, so normalize before the
        # SSM rather than introducing a post-norm variant.
        z = self.norm(x, valid)
        lam, b, c_f, c_r, state_step = self._params(x.dtype, x.device)
        forward = self._run_direction(z, delta, valid, lam, b, c_f, state_step, reverse=False)
        reverse_delta = torch.zeros_like(delta)
        reverse_delta[:, :-1] = delta[:, 1:]
        backward = self._run_direction(z, reverse_delta, valid, lam, b, c_r, state_step, reverse=True)
        y = forward + backward + self.d.to(x.dtype).view(1, 1, -1) * z
        # Speech Commands also keeps the public ``half_glu1`` default:
        # GELU(x), then x * sigmoid(Dense(x)), followed by dropout.
        y = self.dropout(F.gelu(y))
        y = self.dropout(y * torch.sigmoid(self.gate(y)))
        return torch.where(valid.unsqueeze(-1), residual + y, torch.zeros_like(y))


@NETWORKS.register_module("s5_legacy_delta_t")
class S5LegacyDeltaT(nn.Module):
    """S5 dagger baseline: PyTorch ZOH adaptation over legacy timestamp deltas."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        d_model: int = 96,
        state_size: int = 128,
        blocks: int = 16,
        n_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "half_glu1",
        prenorm: bool = True,
        batchnorm: bool = True,
        bidirectional: bool = True,
        **_: object,
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required.")
        if activation != "half_glu1" or not prenorm or not batchnorm or not bidirectional:
            raise ValueError("s5_legacy_delta_t is fixed to the public Speech Commands architecture transfer.")
        self.input = nn.Linear(input_size, d_model)
        nn.init.kaiming_normal_(self.input.weight, nonlinearity="linear")
        nn.init.zeros_(self.input.bias)
        self.layers = nn.ModuleList(
            [S5Layer(d_model, state_size, blocks, dropout) for _ in range(n_layers)]
        )
        self.__output_size = d_model
        self.__hidden_size = d_model

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        valid = valid_tokens(mask, x)
        delta = torch.zeros_like(t)
        delta[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp_min(0.0)
        out = torch.where(valid.unsqueeze(-1), self.input(x), torch.zeros_like(self.input(x)))
        for layer in self.layers:
            out = layer(out, delta, valid)
        return out, out[:, -1, :]

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
