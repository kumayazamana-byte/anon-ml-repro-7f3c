"""Thin UEA classification adapter around pinned upstream t-PatchGNN code."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import torch
from torch import nn


UPSTREAM_REPOSITORY = "https://github.com/usail-hkust/t-PatchGNN.git"
UPSTREAM_COMMIT = "00c94e7bbaf21c71b03ed84ff690ae59e37129e5"
UPSTREAM_SOURCE_SHA256 = {
    "tPatchGNN/model/tPatchGNN.py": "ba9db3896a025103369e27b710794fe7cdce6d03b508c02b5bce5ae12aa6a26c",
    "tPatchGNN/model/Transformer_EncDec.py": "7b26f4459735c341f161011f6bc24b9deb8ecc3bcf61e4f15431998272a44fb4",
    "tPatchGNN/model/SelfAttention_Family.py": "b0539b2893038b1d18ca4c230358dd1f8dd0ae7dc0af7f610af419fbf41b88c6",
}


def canonical_source_sha256(content: bytes) -> str:
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def default_upstream_root() -> Path:
    configured = os.environ.get("TPATCHGNN_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "third_party" / "t-PatchGNN"


def verify_upstream(root: Path) -> dict[str, object]:
    root = root.resolve()
    hashes = {}
    for relative, expected in UPSTREAM_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(
                f"Missing upstream t-PatchGNN source at {path}. "
                "Run scripts/setup_tpatchgnn_upstream.sh first."
            )
        actual = canonical_source_sha256(path.read_bytes())
        if actual != expected:
            raise RuntimeError(f"t-PatchGNN source hash mismatch for {relative}.")
        hashes[relative] = actual
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("t-PatchGNN must be a pinned Git checkout.") from exc
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected t-PatchGNN commit {UPSTREAM_COMMIT}, got {commit}.")
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": commit,
        "source_sha256": hashes,
    }


def load_upstream_module(root: Optional[Path] = None) -> tuple[ModuleType, dict[str, object]]:
    root = (root or default_upstream_root()).resolve()
    provenance = verify_upstream(root)
    model_root = root / "tPatchGNN"
    for path in (model_root, root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module("model.tPatchGNN")
    return module, provenance


def patchify_legacy_batch(
    values: torch.Tensor,
    times: torch.Tensor,
    valid: torch.Tensor,
    max_length: int,
    num_patches: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack every retained legacy row into three fixed original-axis patches."""
    if values.ndim != 3 or times.shape != values.shape[:2] or valid.shape != values.shape[:2]:
        raise ValueError("Expected values [B,T,D] and matching times/valid tensors.")
    if max_length < 1 or num_patches < 1:
        raise ValueError("max_length and num_patches must be positive.")
    batch, padded_length, variables = values.shape
    patch_width = (max_length + num_patches - 1) // num_patches
    original_positions = torch.round(times * padded_length).to(torch.long)
    patch_ids = torch.div(original_positions, patch_width, rounding_mode="floor")
    patch_ids = patch_ids.clamp(0, num_patches - 1)
    offsets = original_positions.remainder(patch_width)
    patched_values = values.new_zeros((batch, num_patches, patch_width, variables))
    patched_times = times.new_zeros((batch, num_patches, patch_width, variables))
    patched_mask = values.new_zeros((batch, num_patches, patch_width, variables))
    for batch_index in range(batch):
        token_indices = torch.where(valid[batch_index])[0]
        patches = patch_ids[batch_index, token_indices]
        slots = offsets[batch_index, token_indices]
        pairs = patches * patch_width + slots
        if pairs.unique().numel() != pairs.numel():
            raise RuntimeError("Two retained rows mapped to the same t-PatchGNN slot.")
        patched_values[batch_index, patches, slots] = values[batch_index, token_indices]
        expanded_times = times[batch_index, token_indices].unsqueeze(-1).expand(-1, variables)
        patched_times[batch_index, patches, slots] = expanded_times
        patched_mask[batch_index, patches, slots] = 1.0
    return patched_values, patched_times, patched_mask


class TPatchGNNLegacyUEAAdapter(nn.Module):
    """Call the upstream encoder and add a documented UEA classification head."""

    def __init__(
        self,
        input_size: int,
        max_length: int,
        num_classes: int,
        num_patches: int = 3,
        hidden_size: int = 64,
        time_embedding_size: int = 10,
        node_embedding_size: int = 10,
        n_heads: int = 1,
        transformer_layers: int = 1,
        graph_layers: int = 1,
        graph_hops: int = 1,
        dropout: float = 0.0,
        upstream_root: Optional[Path] = None,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.max_length = int(max_length)
        self.num_patches = int(num_patches)
        module, self.upstream_provenance = load_upstream_module(upstream_root)
        args = SimpleNamespace(
            device=torch.device("cuda"),
            hid_dim=int(hidden_size),
            ndim=int(input_size),
            npatch=self.num_patches,
            nlayer=int(graph_layers),
            nhead=int(n_heads),
            tf_layer=int(transformer_layers),
            te_dim=int(time_embedding_size),
            node_dim=int(node_embedding_size),
            outlayer="Linear",
            hop=int(graph_hops),
        )
        self.model = module.tPatchGNN(args, dropout=float(dropout))
        self.classifier = nn.Linear(int(hidden_size), int(num_classes))
        self.output_size = int(num_classes)

    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        values, times, mask = patchify_legacy_batch(
            values, times, valid, self.max_length, self.num_patches
        )
        batch, _, patch_length, _ = values.shape
        self.model.batch_size = batch
        values = values.permute(0, 3, 1, 2).reshape(-1, patch_length, 1)
        times = times.permute(0, 3, 1, 2).reshape(-1, patch_length, 1)
        mask = mask.permute(0, 3, 1, 2).reshape(-1, patch_length, 1)
        encoded = torch.cat([values, self.model.LearnableTE(times)], dim=-1)
        per_variable = self.model.IMTS_Model(encoded, mask)
        return self.classifier(per_variable.mean(dim=1))
