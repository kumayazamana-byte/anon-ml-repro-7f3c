"""Thin adapter that calls the pinned upstream Hi-Patch implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import torch
from torch import nn


UPSTREAM_REPOSITORY = "https://github.com/qianlima-lab/Hi-Patch.git"
UPSTREAM_COMMIT = "ae3cb7c672e045db4b11085251a60b0c33b8d0db"
UPSTREAM_MODEL_SHA256 = "075e8c8d2d61e4034f6db35622d70caaac5b261feceda39088aae98b6fce61f0"


def canonical_source_sha256(content: bytes) -> str:
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def default_upstream_root() -> Path:
    configured = os.environ.get("HI_PATCH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "third_party" / "Hi-Patch"


def verify_upstream(root: Path) -> dict[str, str]:
    root = root.resolve()
    model_path = root / "Hi-Patch" / "model" / "hipatch.py"
    if not model_path.is_file():
        raise RuntimeError(
            f"Missing upstream Hi-Patch model at {model_path}. "
            "Run scripts/setup_hi_patch_upstream.sh first."
        )
    digest = canonical_source_sha256(model_path.read_bytes())
    if digest != UPSTREAM_MODEL_SHA256:
        raise RuntimeError(
            f"Hi-Patch model hash mismatch: expected {UPSTREAM_MODEL_SHA256}, got {digest}."
        )
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Hi-Patch must be a pinned Git checkout, not an untracked copy.") from exc
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected Hi-Patch commit {UPSTREAM_COMMIT}, got {commit}.")
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": commit,
        "model_path": str(model_path),
        "model_sha256": digest,
    }


def load_upstream_module(root: Optional[Path] = None) -> tuple[ModuleType, dict[str, str]]:
    root = (root or default_upstream_root()).resolve()
    provenance = verify_upstream(root)
    spec = importlib.util.spec_from_file_location(
        "pinned_hi_patch_upstream", provenance["model_path"]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the pinned Hi-Patch module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module, provenance


def next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def patch_layers(num_patches: int) -> int:
    if num_patches < 1 or num_patches & (num_patches - 1):
        raise ValueError("num_patches must be a positive power of two")
    return num_patches.bit_length()


def patchify_legacy_batch(
    values: torch.Tensor,
    times: torch.Tensor,
    valid: torch.Tensor,
    num_patches: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map retained legacy rows to one original-axis position per patch.

    Legacy MaskTS timestamps are original integer positions divided by the
    current padded retained length. Multiplying by that padded length recovers
    the position used to form deterministic Hi-Patch bins without resampling.
    """
    if values.ndim != 3 or times.shape != values.shape[:2] or valid.shape != values.shape[:2]:
        raise ValueError("Expected values [B,T,D] and matching times/valid tensors.")
    if num_patches < values.shape[1]:
        raise ValueError("num_patches must cover the padded legacy time axis.")
    batch, padded_length, variables = values.shape
    patch_ids = torch.round(times * padded_length).to(torch.long).clamp_(0, num_patches - 1)
    patched_values = values.new_zeros((batch, num_patches, 1, variables))
    patched_times = times.new_zeros((batch, num_patches, 1, variables))
    patched_mask = values.new_zeros((batch, num_patches, 1, variables))
    for batch_index in range(batch):
        token_indices = torch.where(valid[batch_index])[0]
        selected_patches = patch_ids[batch_index, token_indices]
        if selected_patches.unique().numel() != selected_patches.numel():
            raise RuntimeError("Legacy timestamps mapped two retained rows to one Hi-Patch bin.")
        patched_values[batch_index, selected_patches, 0] = values[batch_index, token_indices]
        expanded_times = times[batch_index, token_indices].unsqueeze(-1).expand(-1, variables)
        patched_times[batch_index, selected_patches, 0] = expanded_times
        patched_mask[batch_index, selected_patches, 0] = 1.0
    return patched_values, patched_times, patched_mask


class HiPatchLegacyUEAAdapter(nn.Module):
    """Use upstream ``Hi_Patch`` with frozen legacy MaskTS observations."""

    def __init__(
        self,
        input_size: int,
        max_length: int,
        num_classes: int,
        hidden_size: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        alpha: float = 1.0,
        residual_scale: float = 1.0,
        upstream_root: Optional[Path] = None,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.num_patches = next_power_of_two(int(max_length))
        module, self.upstream_provenance = load_upstream_module(upstream_root)
        args = SimpleNamespace(
            hid_dim=int(hidden_size),
            device=torch.device("cpu"),
            ndim=int(input_size),
            nlayer=int(n_layers),
            nhead=int(n_heads),
            alpha=float(alpha),
            res=float(residual_scale),
            patch_layer=patch_layers(self.num_patches),
            task="classification",
            d_static=0,
            n_class=int(num_classes),
        )
        self.model = module.Hi_Patch(args)
        self.output_size = int(num_classes)

    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        patched = patchify_legacy_batch(values, times, valid, self.num_patches)
        original_batch = values.shape[0]
        if original_batch == 1:
            patched = tuple(tensor.repeat(2, 1, 1, 1) for tensor in patched)
        logits = self.model.classification(patched[0], patched[1], patched[2], None)
        return logits[:original_batch]
