"""Finite-only numerical repair for the three LSST MHA-to-CM retries."""

from __future__ import annotations

import torch

from .hippogp_contiformer import HiPPOGPAttention
from ..module.hipposvgp.kernel import SE_kernel


_ORIGINAL_KERNEL_UU = HiPPOGPAttention._kernel_uu
_ORIGINAL_SE_FORWARD = SE_kernel.forward


def _safe_log_parameter(value: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.nan_to_num(
        value.to(device=device, dtype=dtype), nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)


def stable_se_forward(self, X1: torch.Tensor, X2: torch.Tensor = None, diag: bool = False):
    """Evaluate the existing SE kernel with finite, bounded log parameters."""
    if X2 is None:
        X2 = X1
    log_sf = _safe_log_parameter(self.log_sf, X1.dtype, X1.device)
    log_ls = _safe_log_parameter(self.log_ls, X1.dtype, X1.device)
    X1 = X1 * torch.exp(-log_ls)
    X2 = X2.to(device=X1.device, dtype=X1.dtype) * torch.exp(-log_ls)
    signal = torch.exp(log_sf)
    if diag:
        if X2 is not X1 and X2.shape != X1.shape:
            raise ValueError("Cannot compute diagonal elements for two inputs")
        return signal * torch.exp(-0.5 * (X1 - X2).pow(2).sum(-1))
    if X1.dim() == 3:
        factor1 = X1.pow(2).sum(-1, keepdim=True)
        factor2 = X2.pow(2).sum(-1, keepdim=True)
        cross = X1 @ X2.transpose(-1, -2)
        distance = factor1 + factor2.transpose(-1, -2) - 2.0 * cross
    else:
        factor1 = X1.pow(2).sum(-1)
        factor2 = X2.pow(2).sum(-1)
        distance = factor1.unsqueeze(1) + factor2.unsqueeze(0) - 2.0 * X1 @ X2.T
    return signal * torch.exp(-0.5 * distance.clamp_min(0.0))


def stable_kernel_uu_finite(self, device, dtype):
    """Build a symmetric positive-definite Kuu without changing the CM equation."""
    dense_t, dense_basis = self._dense_basis(device, dtype)
    log_sf = _safe_log_parameter(self.temporal_kernel.log_sf, dtype, device)
    log_ls = _safe_log_parameter(self.temporal_kernel.log_ls, dtype, device)
    scaled_t = dense_t * torch.exp(-log_ls)
    squared_distance = (
        scaled_t.square().sum(-1, keepdim=True)
        + scaled_t.square().sum(-1).unsqueeze(0)
        - 2.0 * scaled_t @ scaled_t.transpose(0, 1)
    ).clamp_min(0.0)
    kernel_tt = torch.exp(log_sf - 0.5 * squared_distance)
    kuu = dense_basis.transpose(0, 1) @ kernel_tt @ dense_basis
    kuu = kuu / float(self.basis_grid_size * self.basis_grid_size)
    kuu = 0.5 * (kuu + kuu.transpose(0, 1))
    scale = kuu.diagonal().abs().mean().detach().clamp_min(1.0)
    floor = (self.jitter * scale).detach().clamp_min(1e-5)
    min_eigenvalue = torch.linalg.eigvalsh(kuu.detach()).amin()
    diagonal_shift = torch.maximum(floor - min_eigenvalue, torch.zeros_like(floor))
    eye = torch.eye(self.hippo_order, device=device, dtype=dtype)
    return kuu + (floor + diagonal_shift) * eye


def install_finite_repair() -> None:
    if HiPPOGPAttention._kernel_uu is not stable_kernel_uu_finite:
        HiPPOGPAttention._kernel_uu = stable_kernel_uu_finite
    if SE_kernel.forward is not stable_se_forward:
        SE_kernel.forward = stable_se_forward


def uninstall_finite_repair() -> None:
    HiPPOGPAttention._kernel_uu = _ORIGINAL_KERNEL_UU
    SE_kernel.forward = _ORIGINAL_SE_FORWARD
