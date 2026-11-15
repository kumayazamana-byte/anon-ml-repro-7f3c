"""Numerically stabilized Kuu construction for isolated LSST repair runs."""

from __future__ import annotations

import torch

from .hippogp_contiformer import HiPPOGPAttention


_ORIGINAL_KERNEL_UU = HiPPOGPAttention._kernel_uu


def stable_kernel_uu(self, device, dtype):
    dense_t, dense_basis = self._dense_basis(device, dtype)
    log_sf = torch.nan_to_num(
        self.temporal_kernel.log_sf.to(dtype=dtype), nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    log_ls = torch.nan_to_num(
        self.temporal_kernel.log_ls.to(dtype=dtype), nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
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
    eye = torch.eye(self.hippo_order, device=device, dtype=dtype)
    return kuu + self.jitter * scale * eye


def install_numerical_repair() -> None:
    if HiPPOGPAttention._kernel_uu is not stable_kernel_uu:
        HiPPOGPAttention._kernel_uu = stable_kernel_uu


def uninstall_numerical_repair() -> None:
    HiPPOGPAttention._kernel_uu = _ORIGINAL_KERNEL_UU
