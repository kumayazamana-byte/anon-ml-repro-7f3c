# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .tsrnn import NETWORKS
from .contiformer import PositionwiseFeedForward
from ..module.hipposvgp import HiPPO_LegS, SE_kernel


PAD = 0


class RMSMemoryNorm(nn.Module):
    """RMSNorm over the HiPPO memory block without mean-centering."""

    def __init__(self, normalized_shape, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        rms = x.pow(2).mean(dim=(-2, -1), keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def shifted_legendre_basis(time: torch.Tensor, order: int) -> torch.Tensor:
    """Evaluate normalized shifted Legendre basis on [0, 1].

    Args:
        time: Tensor with shape [...], assumed normalized into [0, 1].
        order: Number of basis functions.

    Returns:
        Tensor with shape [..., order].
    """
    x = 2.0 * time.clamp(0.0, 1.0) - 1.0
    basis = []
    p0 = torch.ones_like(x)
    basis.append(p0)
    if order > 1:
        p1 = x
        basis.append((3.0 ** 0.5) * p1)
        p_nm2, p_nm1 = p0, p1
        for n in range(2, order):
            p_n = ((2 * n - 1) * x * p_nm1 - (n - 1) * p_nm2) / n
            basis.append(((2 * n + 1) ** 0.5) * p_n)
            p_nm2, p_nm1 = p_nm1, p_n
    return torch.stack(basis, dim=-1)


class CoordinateKNNResidual(nn.Module):
    """Coordinate kNN local residual for sparse-to-dense query layouts."""

    def __init__(
            self,
            d_model: int,
            n_head: int,
            d_v: int,
            local_knn_k: int = 8,
            local_residual_init: float = 0.1,
            dropout: float = 0.1,
            use_relpos: bool = False,
    ):
        super().__init__()
        if local_knn_k < 1:
            raise ValueError("local_knn_k must be >= 1.")
        self.d_model = d_model
        self.n_head = n_head
        self.d_v = d_v
        self.local_knn_k = int(local_knn_k)
        self.use_relpos = use_relpos

        self.coord_mlp = nn.Sequential(
            nn.LazyLinear(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.q_proj = nn.Linear(d_model, n_head * d_v)
        self.k_proj = nn.Linear(2 * d_model, n_head * d_v)
        self.v_proj = nn.Linear(d_model, n_head * d_v)
        self.out_proj = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.local_scale = nn.Parameter(torch.tensor(float(local_residual_init)))
        self.raw_tau = nn.Parameter(torch.zeros(n_head))
        if use_relpos:
            self.rel_mlp = nn.Sequential(
                nn.LazyLinear(d_model),
                nn.GELU(),
                nn.Linear(d_model, n_head),
            )
        else:
            self.rel_mlp = None
        self.last_debug = {}

    @staticmethod
    def as_coords(coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 2:
            return coords.unsqueeze(-1)
        if coords.dim() == 3:
            return coords
        raise ValueError("coordinates must have shape [B, N] or [B, N, C].")

    @staticmethod
    def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, query_len, k = indices.shape
        flat_idx = indices.reshape(batch, query_len * k)
        gather_idx = flat_idx.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        gathered = torch.gather(values, dim=1, index=gather_idx)
        return gathered.reshape(batch, query_len, k, values.shape[-1])

    def nearest_context(
            self,
            x: torch.Tensor,
            input_coords: torch.Tensor,
            query_coords: torch.Tensor,
            valid: torch.Tensor,
    ) -> torch.Tensor:
        input_coords = self.as_coords(input_coords).to(dtype=x.dtype, device=x.device)
        query_coords = self.as_coords(query_coords).to(dtype=x.dtype, device=x.device)
        dist = torch.cdist(query_coords, input_coords, p=2).pow(2)
        dist = dist.masked_fill(~valid.unsqueeze(1), float("inf"))
        nearest = dist.argmin(dim=-1)
        return self._gather_neighbors(x, nearest.unsqueeze(-1)).squeeze(2)

    def forward(
            self,
            x: torch.Tensor,
            input_coords: torch.Tensor,
            query_coords: torch.Tensor,
            valid: torch.Tensor,
    ) -> torch.Tensor:
        input_coords = self.as_coords(input_coords).to(dtype=x.dtype, device=x.device)
        query_coords = self.as_coords(query_coords).to(dtype=x.dtype, device=x.device)
        batch, query_len, coord_dim = query_coords.shape
        k_eff = min(self.local_knn_k, input_coords.shape[1])

        dist = torch.cdist(query_coords, input_coords, p=2).pow(2)
        dist = dist.masked_fill(~valid.unsqueeze(1), float("inf"))
        knn_dist, knn_idx = torch.topk(dist, k=k_eff, dim=-1, largest=False)
        neighbor_valid = torch.gather(
            valid.unsqueeze(1).expand(-1, query_len, -1), dim=-1, index=knn_idx
        )

        coord_in = self.coord_mlp(input_coords)
        coord_q = self.coord_mlp(query_coords)
        neighbor_x = self._gather_neighbors(x, knn_idx)
        neighbor_coord = self._gather_neighbors(coord_in, knn_idx)
        neighbor_raw_coord = self._gather_neighbors(input_coords, knn_idx)

        q = self.q_proj(coord_q).view(batch, query_len, self.n_head, self.d_v)
        kv_input = torch.cat(
            [neighbor_x, neighbor_coord], dim=-1
        ).reshape(batch * query_len * k_eff, 2 * self.d_model)
        keys = self.k_proj(kv_input).view(batch, query_len, k_eff, self.n_head, self.d_v)
        values = self.v_proj(neighbor_x).view(batch, query_len, k_eff, self.n_head, self.d_v)

        scores = torch.einsum("bqhd,bqkhd->bqkh", q, keys) / math.sqrt(float(self.d_v))
        tau = F.softplus(self.raw_tau).to(dtype=x.dtype) + 1e-4
        scores = scores - knn_dist.unsqueeze(-1) / (2.0 * tau.view(1, 1, 1, -1).pow(2))
        rel_bias_mean = None
        if self.rel_mlp is not None:
            delta = query_coords.unsqueeze(2) - neighbor_raw_coord
            rel_bias = self.rel_mlp(delta.reshape(batch * query_len * k_eff, coord_dim))
            rel_bias = rel_bias.view(batch, query_len, k_eff, self.n_head)
            scores = scores + rel_bias
            rel_bias_mean = rel_bias.detach().mean().item()

        scores = scores.masked_fill(~neighbor_valid.unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=2)
        weights = torch.where(neighbor_valid.unsqueeze(-1), weights, torch.zeros_like(weights))
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        weights = weights / denom

        local = torch.einsum("bqkh,bqkhd->bqhd", weights, values)
        local = local.reshape(batch, query_len, self.n_head * self.d_v)
        local = self.out_proj(local)
        self.last_debug = {
            "active": True,
            "type": "knn_relpos" if self.use_relpos else "knn",
            "lambda_kNN": float(self.local_scale.detach().cpu()),
            "tau": [float(v) for v in tau.detach().cpu()],
            "k": int(k_eff),
            "mean_relative_bias": rel_bias_mean,
        }
        return self.dropout(local)


class HiPPOGPAttention(nn.Module):
    """HiPPO interdomain GP attention.

    This is the mean-only operator from the draft:
        O(R) = K_Ru (K_uu + eps I)^-1 M_u,
        M_u = normalized K_Tu^T V.

    Two memory backends are exposed for ablation:
      - analytic: builds data-conditioned memory via analytic Legendre features.
      - ode: uses the supplied HiPPO_LegS recurrence to project value streams.
    """

    def __init__(
            self,
            d_model: int,
            n_head: int,
            d_v: int,
            hippo_order: int = 16,
            basis_backend: str = "analytic",
            dropout: float = 0.1,
            local_residual: bool = True,
            local_residual_type: str = "self_attn",
            local_knn_k: int = 8,
            local_residual_init: float = 0.1,
            uncertainty_gate: bool = False,
            mixture_type: str = "default",
            memory_normalization: str = "l1",
            gp_rope: str = "none",
            gp_rope_base: float = 10000.0,
            jitter: float = 1e-4,
            basis_grid_size: int = 128,
            max_length: int = 2048,
            hippo_fast: bool = False,
    ):
        super().__init__()
        if d_model != n_head * d_v:
            raise ValueError("d_model must equal n_head * d_v for HiPPOGPAttention.")
        if basis_backend not in {"analytic", "ode"}:
            raise ValueError("basis_backend must be 'analytic' or 'ode'.")
        if memory_normalization not in {"none", "sum", "mean", "l1", "l2", "quadrature", "layernorm", "rmsnorm"}:
            raise ValueError(
                "memory_normalization must be one of: none, sum, mean, l1, l2, quadrature, layernorm, rmsnorm."
            )
        if gp_rope not in {"none", "fixed", "learned"}:
            raise ValueError("gp_rope must be one of: none, fixed, learned.")
        if local_residual_type not in {"self_attn", "knn", "knn_relpos"}:
            raise ValueError("local_residual_type must be one of: self_attn, knn, knn_relpos.")
        if mixture_type not in {"default", "symmetric"}:
            raise ValueError("mixture_type must be 'default' or 'symmetric'.")

        self.d_model = d_model
        self.n_head = n_head
        self.d_v = d_v
        self.hippo_order = hippo_order
        self.basis_backend = basis_backend
        self.memory_normalization = memory_normalization
        self.gp_rope = gp_rope
        self.jitter = jitter
        self.basis_grid_size = basis_grid_size
        self.hippo_fast = hippo_fast

        self.value_proj = nn.Linear(d_model, n_head * d_v)
        self.out_proj = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        self.temporal_kernel = SE_kernel(input_dim=1, device="cpu")
        self.temporal_kernel.float()

        self.dropout = nn.Dropout(dropout)
        self.local_residual = local_residual
        self.local_residual_type = local_residual_type
        self.local_knn_k = int(local_knn_k)
        self.mixture_type = mixture_type
        self.local_debug = {"active": False, "type": local_residual_type}
        if local_residual:
            if local_residual_type == "self_attn":
                self.local_attn = nn.MultiheadAttention(
                    d_model, n_head, dropout=dropout, batch_first=True
                )
                self.local_scale = nn.Parameter(torch.tensor(float(local_residual_init)))
                self.knn_residual = None
            else:
                self.local_attn = None
                self.local_scale = None
                self.knn_residual = CoordinateKNNResidual(
                    d_model=d_model,
                    n_head=n_head,
                    d_v=d_v,
                    local_knn_k=local_knn_k,
                    local_residual_init=local_residual_init,
                    dropout=dropout,
                    use_relpos=local_residual_type == "knn_relpos",
                )
        else:
            self.local_attn = None
            self.local_scale = None
            self.knn_residual = None

        self.uncertainty_gate = uncertainty_gate
        if uncertainty_gate:
            self.gate = nn.Linear(d_model + 1, d_model)
            self.coord_gate_embed = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            ) if mixture_type == "symmetric" else None
        else:
            self.gate = None
            self.coord_gate_embed = None

        if gp_rope != "none":
            num_pairs = hippo_order // 2
            inv_freq = gp_rope_base ** (
                -torch.arange(0, num_pairs, dtype=torch.float32) / max(num_pairs, 1)
            )
            if gp_rope == "learned":
                self.gp_rope_log_freq = nn.Parameter(inv_freq.clamp_min(1e-8).log())
            else:
                self.register_buffer("gp_rope_inv_freq", inv_freq, persistent=False)
                self.gp_rope_log_freq = None
        else:
            self.gp_rope_log_freq = None

        if memory_normalization == "layernorm":
            self.memory_output_norm = nn.LayerNorm((hippo_order, d_v))
        elif memory_normalization == "rmsnorm":
            self.memory_output_norm = RMSMemoryNorm((hippo_order, d_v))
        else:
            self.memory_output_norm = None

        self.hippo = HiPPO_LegS(
            hippo_order, device="cpu", max_length=max_length, measure="legs"
        )

    def _ensure_hippo_device(self, device, dtype):
        self.hippo.A_stacked = self.hippo.A_stacked.to(device=device, dtype=dtype)
        self.hippo.B_stacked = self.hippo.B_stacked.to(device=device, dtype=dtype)
        self.hippo.eval_matrix = self.hippo.eval_matrix.to(device=device, dtype=dtype)

    @staticmethod
    def _valid_from_mask(mask, x):
        if mask is None:
            return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if mask.dim() == 3:
            return ~mask.all(dim=-1)
        if mask.dim() == 2:
            return ~mask
        raise ValueError("mask must have shape [B, T] or [B, T, T].")

    @staticmethod
    def _time_bounds(t, valid):
        t = t.float()
        t = torch.where(valid, t, torch.zeros_like(t))
        t_min = torch.where(valid, t, torch.full_like(t, float("inf"))).amin(dim=1, keepdim=True)
        t_min = torch.where(torch.isfinite(t_min), t_min, torch.zeros_like(t_min))
        t_shift = torch.where(valid, t - t_min, torch.zeros_like(t))
        t_scale = t_shift.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return t_min, t_scale

    @staticmethod
    def _apply_time_bounds(t, valid, t_min, t_scale, clamp=True):
        normalized = (t.float() - t_min) / t_scale
        normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
        return normalized.clamp(0.0, 1.0) if clamp else normalized

    @classmethod
    def _normalize_time(cls, t, valid):
        t_min, t_scale = cls._time_bounds(t, valid)
        return cls._apply_time_bounds(t, valid, t_min, t_scale, clamp=True)

    def _dense_basis(self, device, dtype):
        dense_t = torch.linspace(0.0, 1.0, self.basis_grid_size, device=device, dtype=dtype)
        dense_basis = shifted_legendre_basis(dense_t, self.hippo_order)
        return dense_t.reshape(-1, 1), dense_basis

    def _interdomain_features(self, t_norm):
        batch, length = t_norm.shape
        device, dtype = t_norm.device, t_norm.dtype
        dense_t, dense_basis = self._dense_basis(device, dtype)
        dense_batch = dense_t.unsqueeze(0).expand(batch, -1, -1)
        query = t_norm.unsqueeze(-1)
        kernel_rt = self.temporal_kernel(query, dense_batch)
        return kernel_rt @ dense_basis / float(self.basis_grid_size)

    def _apply_gp_rope(self, features, t_norm):
        if self.gp_rope == "none" or self.hippo_order < 2:
            return features

        pair_dim = (self.hippo_order // 2) * 2
        paired = features[..., :pair_dim].reshape(*features.shape[:-1], -1, 2)
        if self.gp_rope == "learned":
            freq = self.gp_rope_log_freq.exp().to(device=features.device, dtype=features.dtype)
        else:
            freq = self.gp_rope_inv_freq.to(device=features.device, dtype=features.dtype)
        angle = (2.0 * torch.pi * t_norm.unsqueeze(-1)) * freq.view(1, 1, -1)
        cos, sin = angle.cos(), angle.sin()
        first, second = paired[..., 0], paired[..., 1]
        rotated = torch.stack(
            [first * cos - second * sin, first * sin + second * cos], dim=-1
        ).reshape(*features.shape[:-1], pair_dim)
        if pair_dim == self.hippo_order:
            return rotated
        return torch.cat([rotated, features[..., pair_dim:]], dim=-1)

    def _kernel_uu(self, device, dtype):
        dense_t, dense_basis = self._dense_basis(device, dtype)
        kernel_tt = self.temporal_kernel(dense_t, dense_t)
        kuu = dense_basis.transpose(0, 1) @ kernel_tt @ dense_basis
        kuu = kuu / float(self.basis_grid_size * self.basis_grid_size)
        eye = torch.eye(self.hippo_order, device=device, dtype=dtype)
        return kuu + self.jitter * eye

    @staticmethod
    def _quadrature_weights(t_norm, valid):
        t_masked = torch.where(valid, t_norm, torch.zeros_like(t_norm))
        left = t_masked - torch.cat([t_masked[:, :1], t_masked[:, :-1]], dim=1)
        right = torch.cat([t_masked[:, 1:], t_masked[:, -1:]], dim=1) - t_masked
        weights = 0.5 * (left.clamp_min(0.0) + right.clamp_min(0.0))
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _normalize_features(self, features, valid, t_norm):
        features = torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))
        if self.memory_normalization in {"none", "sum"}:
            return features
        if self.memory_normalization == "mean":
            count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(features.dtype)
            return features / count.unsqueeze(-1)
        if self.memory_normalization == "quadrature":
            weights = self._quadrature_weights(t_norm, valid).unsqueeze(-1).to(features.dtype)
            return features * weights
        if self.memory_normalization in {"layernorm", "rmsnorm"}:
            return features
        denom = features.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        if self.memory_normalization == "l2":
            denom = torch.sqrt((features ** 2).sum(dim=1, keepdim=True).clamp_min(1e-6))
        return features / denom

    def _analytic_memory(self, values, t_norm, valid):
        features = self._apply_gp_rope(self._interdomain_features(t_norm), t_norm)
        features = self._normalize_features(features, valid, t_norm)
        return torch.einsum("bnm,bhnd->bhmd", features, values)

    def _ode_memory(self, values, valid):
        self._ensure_hippo_device(values.device, values.dtype)
        values = torch.where(valid[:, None, :, None], values, torch.zeros_like(values))
        stream = values.permute(2, 0, 1, 3).contiguous()
        coeffs = self.hippo(stream, prev_discrete_steps=0, ini=None, fast=self.hippo_fast)
        return coeffs[-1].permute(0, 1, 3, 2).contiguous()

    def forward(self, x, t, mask=None, query_t=None, time_bounds=None):
        valid = self._valid_from_mask(mask, x)
        if query_t is None:
            query_t = t
            query_valid = valid
        else:
            query_valid = torch.ones(query_t.shape, dtype=torch.bool, device=x.device)

        if time_bounds is None:
            t_norm = self._normalize_time(t, valid)
            query_t_norm = self._normalize_time(query_t, query_valid)
        else:
            support_min, support_scale = time_bounds
            t_norm = self._apply_time_bounds(t, valid, support_min, support_scale, clamp=True)
            # Query coordinates share the observed support frame.  Do not clamp
            # legal future queries beyond the observed maximum.
            query_t_norm = self._apply_time_bounds(
                query_t, query_valid, support_min, support_scale, clamp=False
            )

        values = self.value_proj(x).view(x.size(0), x.size(1), self.n_head, self.d_v)
        values = values.permute(0, 2, 1, 3).contiguous()

        if self.basis_backend == "ode":
            memory = self._ode_memory(values, valid)
        else:
            memory = self._analytic_memory(values, t_norm, valid)
        if self.memory_output_norm is not None:
            memory = self.memory_output_norm(memory)

        kuu = self._kernel_uu(x.device, x.dtype)
        chol = torch.linalg.cholesky(kuu)
        flat_memory = memory.reshape(-1, self.hippo_order, self.d_v)
        solved_memory = torch.cholesky_solve(flat_memory, chol).view_as(memory)

        query_features = self._apply_gp_rope(
            self._interdomain_features(query_t_norm), query_t_norm
        )
        gp_out = torch.einsum("bqm,bhmd->bqhd", query_features, solved_memory)
        gp_out = gp_out.reshape(x.size(0), query_t.shape[1], self.n_head * self.d_v)
        gp_out = self.out_proj(gp_out)

        local_out = None
        self.local_debug = {"active": False, "type": self.local_residual_type}
        if self.local_residual:
            if self.local_residual_type == "self_attn" and query_t.shape[1] == x.shape[1]:
                safe_valid = valid.clone()
                empty = ~safe_valid.any(dim=1)
                if empty.any():
                    safe_valid[empty, 0] = True
                key_padding_mask = ~safe_valid
                local_out, _ = self.local_attn(x, x, x, key_padding_mask=key_padding_mask)
                local_out = torch.where(valid.unsqueeze(-1), local_out, torch.zeros_like(local_out))
                self.local_debug = {
                    "active": True,
                    "type": "self_attn",
                    "lambda_local": float(self.local_scale.detach().cpu()),
                }
            elif self.local_residual_type in {"knn", "knn_relpos"}:
                local_out = self.knn_residual(x, t_norm, query_t_norm, valid)
                self.local_debug = dict(self.knn_residual.last_debug)

        if self.uncertainty_gate:
            solved_basis = torch.cholesky_solve(query_features.transpose(1, 2), chol)
            explained = (query_features.transpose(1, 2) * solved_basis).sum(dim=1)
            variance = (torch.exp(self.temporal_kernel.log_sf).to(x.dtype) - explained).clamp_min(0.0)
            if self.mixture_type == "symmetric":
                gate_context = self.coord_gate_embed(query_t_norm.unsqueeze(-1).to(dtype=x.dtype))
            elif query_t.shape[1] == x.shape[1]:
                gate_context = x[:, :query_t.shape[1], :]
            elif self.local_residual and self.knn_residual is not None:
                gate_context = self.knn_residual.nearest_context(x, t_norm, query_t_norm, valid)
            else:
                gate_context = torch.zeros(
                    x.size(0), query_t.shape[1], x.size(-1), device=x.device, dtype=x.dtype
                )
            gate_input = torch.cat([gate_context, variance.unsqueeze(-1)], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            if self.mixture_type == "symmetric" and local_out is not None:
                gp_out = gate * gp_out + (1.0 - gate) * local_out
                self.local_debug.update({
                    "mixture_type": "symmetric",
                    "alpha_mean": float(gate.detach().mean().cpu()),
                })
            else:
                gp_out = gate * gp_out

        if local_out is not None and not (self.uncertainty_gate and self.mixture_type == "symmetric"):
            if self.local_residual_type == "self_attn":
                gp_out = gp_out + self.local_scale * local_out
            else:
                gp_out = gp_out + self.knn_residual.local_scale * local_out

        gp_out = torch.where(query_valid.unsqueeze(-1), gp_out, torch.zeros_like(gp_out))
        return self.dropout(gp_out)


class MhaToCmHiPPOGPAttention(nn.Module):
    """The MHA -> coordinate-memory order used by the Spiral operator audit."""

    def __init__(
            self,
            d_model: int,
            n_head: int,
            d_v: int,
            dropout: float = 0.1,
            local_residual_init: float = 0.1,
            **attention_kwargs,
    ):
        super().__init__()
        if not attention_kwargs.pop("local_residual", True):
            raise ValueError("mha_to_cm requires an active V3 local residual.")
        if attention_kwargs.get("uncertainty_gate", False):
            raise ValueError("mha_to_cm does not support an uncertainty gate.")
        if attention_kwargs.get("local_residual_type", "self_attn") != "self_attn":
            raise ValueError("mha_to_cm requires the V3 self-attention local branch.")
        self.cm_attn = HiPPOGPAttention(
            d_model=d_model,
            n_head=n_head,
            d_v=d_v,
            dropout=dropout,
            local_residual=False,
            **attention_kwargs,
        )
        self.local_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.local_scale = nn.Parameter(torch.tensor(float(local_residual_init)))

    def forward(self, x, t, mask=None, query_t=None, time_bounds=None):
        if query_t is not None and query_t.shape[1] != x.shape[1]:
            raise ValueError("mha_to_cm only supports token-aligned coordinate queries.")
        valid = HiPPOGPAttention._valid_from_mask(mask, x)
        # PyTorch's eval fast path returns NaNs when every key is masked in a
        # row.  Keep the row semantically padded while unmasking one dummy key
        # for the attention kernel; the result is zeroed immediately after.
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if empty.any():
            safe_valid[empty, 0] = True
        local, _ = self.local_attn(x, x, x, key_padding_mask=~safe_valid, need_weights=False)
        local = torch.where(valid.unsqueeze(-1), local, torch.zeros_like(local))
        local = self.local_scale * local
        return local + self.cm_attn(
            x + local, t, mask=mask, query_t=query_t, time_bounds=time_bounds
        )


class HiPPOGPEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            d_inner,
            n_head,
            d_v,
            dropout=0.1,
            normalize_before=False,
            **attention_kwargs,
    ):
        super().__init__()
        self.normalize_before = normalize_before
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        operator_order = attention_kwargs.pop("operator_order", "parallel")
        if operator_order == "parallel":
            self.slf_attn = HiPPOGPAttention(
                d_model=d_model,
                n_head=n_head,
                d_v=d_v,
                dropout=dropout,
                **attention_kwargs,
            )
        elif operator_order == "mha_to_cm":
            self.slf_attn = MhaToCmHiPPOGPAttention(
                d_model=d_model,
                n_head=n_head,
                d_v=d_v,
                dropout=dropout,
                **attention_kwargs,
            )
        else:
            raise ValueError("operator_order must be 'parallel' or 'mha_to_cm'.")
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(self, x, t, mask=None, time_bounds=None):
        residual = x
        attn_input = self.layer_norm(x) if self.normalize_before else x
        out = residual + self.slf_attn(attn_input, t, mask=mask, time_bounds=time_bounds)
        if not self.normalize_before:
            out = self.layer_norm(out)
        out = self.pos_ffn(out)
        return out


class HiPPOGPEncoder(nn.Module):
    def __init__(
            self,
            d_model,
            d_inner,
            n_layers,
            n_head,
            d_v,
            dropout,
            normalize_before=False,
            **attention_kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HiPPOGPEncoderLayer(
                    d_model=d_model,
                    d_inner=d_inner,
                    n_head=n_head,
                    d_v=d_v,
                    dropout=dropout,
                    normalize_before=normalize_before,
                    **attention_kwargs,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, t, mask=None, time_bounds=None):
        for layer in self.layers:
            x = layer(x, t, mask=mask, time_bounds=time_bounds)
        return x


class _BaseHiPPOGPContiFormer(nn.Module):
    def __init__(
            self,
            input_size: Optional[int] = None,
            d_model: Optional[int] = 32,
            d_inner: Optional[int] = 128,
            n_layers: Optional[int] = 1,
            n_head: Optional[int] = 4,
            d_k: Optional[int] = 8,
            d_v: Optional[int] = 8,
            dropout: Optional[float] = 0.1,
            hippo_order: Optional[int] = 16,
            basis_backend: Optional[str] = "analytic",
            basis_grid_size: Optional[int] = 128,
            memory_normalization: Optional[str] = "l1",
            local_residual: Optional[bool] = True,
            local_residual_type: Optional[str] = "self_attn",
            local_knn_k: Optional[int] = 8,
            local_residual_init: Optional[float] = 0.1,
            uncertainty_gate: Optional[bool] = False,
            mixture_type: Optional[str] = "default",
            gp_rope: Optional[str] = "none",
            gp_rope_base: Optional[float] = 10000.0,
            jitter: Optional[float] = 1e-4,
            operator_order: Optional[str] = "parallel",
            normalize_before: Optional[bool] = False,
            max_length: int = 100,
            **unused_contiformer_kwargs,
    ):
        super().__init__()
        if d_model != n_head * d_v:
            d_v = d_model // n_head
        if d_model != n_head * d_v:
            raise ValueError("d_model must be divisible by n_head.")

        self.linear = nn.Linear(input_size, d_model) if input_size is not None else None
        self.encoder = HiPPOGPEncoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_v=d_v,
            dropout=dropout,
            normalize_before=normalize_before,
            hippo_order=hippo_order,
            basis_backend=basis_backend,
            basis_grid_size=basis_grid_size,
            memory_normalization=memory_normalization,
            local_residual=local_residual,
            local_residual_type=local_residual_type,
            local_knn_k=local_knn_k,
            local_residual_init=local_residual_init,
            uncertainty_gate=uncertainty_gate,
            mixture_type=mixture_type,
            gp_rope=gp_rope,
            gp_rope_base=gp_rope_base,
            jitter=jitter,
            operator_order=operator_order,
            max_length=max(max_length, 2048),
        )
        self.__output_size = d_model
        self.__hidden_size = d_model

    def forward(self, x, t=None, mask=None):
        if t is None:
            t = torch.linspace(0, 1, x.shape[1], device=x.device)
            t = t.unsqueeze(0).repeat(x.shape[0], 1)
        if self.linear is not None:
            x = self.linear(x)
        enc_output = self.encoder(x, t, mask=mask)
        return enc_output, enc_output[:, -1, :]

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module("hippogp_contiformer_analytic")
class HiPPOGPContiFormerAnalytic(_BaseHiPPOGPContiFormer):
    def __init__(
            self,
            input_size: Optional[int] = None,
            d_model: Optional[int] = 32,
            d_inner: Optional[int] = 128,
            n_layers: Optional[int] = 1,
            n_head: Optional[int] = 4,
            d_k: Optional[int] = 8,
            d_v: Optional[int] = 8,
            dropout: Optional[float] = 0.1,
            hippo_order: Optional[int] = 16,
            basis_grid_size: Optional[int] = 128,
            memory_normalization: Optional[str] = "l1",
            local_residual: Optional[bool] = True,
            local_residual_type: Optional[str] = "self_attn",
            local_knn_k: Optional[int] = 8,
            local_residual_init: Optional[float] = 0.1,
            uncertainty_gate: Optional[bool] = False,
            mixture_type: Optional[str] = "default",
            gp_rope: Optional[str] = "none",
            gp_rope_base: Optional[float] = 10000.0,
            jitter: Optional[float] = 1e-4,
            operator_order: Optional[str] = "parallel",
            normalize_before: Optional[bool] = False,
            max_length: int = 100,
    ):
        super().__init__(
            input_size=input_size,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            hippo_order=hippo_order,
            basis_backend="analytic",
            basis_grid_size=basis_grid_size,
            memory_normalization=memory_normalization,
            local_residual=local_residual,
            local_residual_type=local_residual_type,
            local_knn_k=local_knn_k,
            local_residual_init=local_residual_init,
            uncertainty_gate=uncertainty_gate,
            mixture_type=mixture_type,
            gp_rope=gp_rope,
            gp_rope_base=gp_rope_base,
            jitter=jitter,
            operator_order=operator_order,
            normalize_before=normalize_before,
            max_length=max_length,
        )


@NETWORKS.register_module("hippogp_contiformer_ode")
class HiPPOGPContiFormerODE(_BaseHiPPOGPContiFormer):
    def __init__(
            self,
            input_size: Optional[int] = None,
            d_model: Optional[int] = 32,
            d_inner: Optional[int] = 128,
            n_layers: Optional[int] = 1,
            n_head: Optional[int] = 4,
            d_k: Optional[int] = 8,
            d_v: Optional[int] = 8,
            dropout: Optional[float] = 0.1,
            hippo_order: Optional[int] = 16,
            basis_grid_size: Optional[int] = 128,
            memory_normalization: Optional[str] = "l1",
            local_residual: Optional[bool] = True,
            local_residual_type: Optional[str] = "self_attn",
            local_knn_k: Optional[int] = 8,
            local_residual_init: Optional[float] = 0.1,
            uncertainty_gate: Optional[bool] = False,
            mixture_type: Optional[str] = "default",
            gp_rope: Optional[str] = "none",
            gp_rope_base: Optional[float] = 10000.0,
            jitter: Optional[float] = 1e-4,
            operator_order: Optional[str] = "parallel",
            normalize_before: Optional[bool] = False,
            max_length: int = 100,
    ):
        super().__init__(
            input_size=input_size,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            hippo_order=hippo_order,
            basis_backend="ode",
            basis_grid_size=basis_grid_size,
            memory_normalization=memory_normalization,
            local_residual=local_residual,
            local_residual_type=local_residual_type,
            local_knn_k=local_knn_k,
            local_residual_init=local_residual_init,
            uncertainty_gate=uncertainty_gate,
            mixture_type=mixture_type,
            gp_rope=gp_rope,
            gp_rope_base=gp_rope_base,
            jitter=jitter,
            operator_order=operator_order,
            normalize_before=normalize_before,
            max_length=max_length,
        )
