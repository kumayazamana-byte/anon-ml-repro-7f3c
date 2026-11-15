"""Thin adapter for the pinned upstream MambaSL implementation."""

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


UPSTREAM_REPOSITORY = "https://github.com/yoom618/MambaSL.git"
UPSTREAM_COMMIT = "ba41e4b0be1bd3909b71b3fc4fae02edf087f5f9"
UPSTREAM_SOURCE_SHA256 = {
    "models/MambaSingleLayer.py": "1cb2068ebbd17c184299df293d2d489d5cdb1ddacb3e9927d09c5d3bf8f0c90f",
    "layers/MambaBlock.py": "cbcbbd6cbb83366e40ffe5926da16b4ca0aa11b6649e2259750994b60527a390",
    "layers/Embed.py": "1bc5b455ada8aa01ba1f9d8f0dad5676ee026b5dc04dbf2ef166e91a3f3a17f4",
    "layers/RevIN.py": "1e08d0b3679f4ef79ac7f98a7f719f8d7f8a04595022ed34f702c7d70c0c4d90",
}


def canonical_source_sha256(content: bytes) -> str:
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def default_upstream_root() -> Path:
    configured = os.environ.get("MAMBASL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "third_party" / "MambaSL"


def verify_upstream(root: Path) -> dict[str, object]:
    root = root.resolve()
    observed = {}
    for relative, expected in UPSTREAM_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(
                f"Missing upstream MambaSL source at {path}. "
                "Run scripts/setup_mambasl_upstream.sh first."
            )
        digest = canonical_source_sha256(path.read_bytes())
        if digest != expected:
            raise RuntimeError(
                f"MambaSL source hash mismatch for {relative}: expected {expected}, got {digest}."
            )
        observed[relative] = digest
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("MambaSL must be a pinned Git checkout.") from exc
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected MambaSL commit {UPSTREAM_COMMIT}, got {commit}.")
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": commit,
        "source_sha256": observed,
    }


def load_upstream_module(root: Optional[Path] = None) -> tuple[ModuleType, dict[str, object]]:
    root = (root or default_upstream_root()).resolve()
    provenance = verify_upstream(root)
    model_path = root / "models" / "MambaSingleLayer.py"
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("pinned_mambasl_upstream", model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load the pinned MambaSL module.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(root))
    return module, provenance


def official_kernel_size(max_length: int) -> int:
    """Use the public MambaSL sequence-length rule, without outcome tuning."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    return max(3, (int(max_length) + 49) // 50)


class MambaSLLegacyUEAAdapter(nn.Module):
    """Call upstream MambaSL on rows retained by legacy MaskTS."""

    def __init__(
        self,
        input_size: int,
        max_length: int,
        num_classes: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        n_heads: int = 8,
        dropout: float = 0.1,
        tv_dt: bool = True,
        tv_B: bool = True,
        tv_C: bool = True,
        use_D: bool = False,
        upstream_root: Optional[Path] = None,
    ) -> None:
        super().__init__()
        module, self.upstream_provenance = load_upstream_module(upstream_root)
        self.num_kernels = official_kernel_size(max_length)
        config = SimpleNamespace(
            task_name="classification",
            seq_len=int(max_length),
            label_len=0,
            pred_len=0,
            c_out=int(num_classes),
            dropout=float(dropout),
            num_kernels=self.num_kernels,
            save_csv=False,
            use_revin=False,
            enc_in=int(input_size),
            d_model=int(d_model),
            embed="timeF",
            freq="h",
            d_ff=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            tv_dt=int(tv_dt),
            tv_B=int(tv_B),
            tv_C=int(tv_C),
            use_D=int(use_D),
            device=None,
            mamba_projection_type="gating",
            n_heads=int(n_heads),
            num_class=int(num_classes),
        )
        self.model = module.Model(config)
        self.output_size = int(num_classes)

    def forward(
        self,
        values: torch.Tensor,
        legacy_timestamps: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        del legacy_timestamps
        return self.model(values, valid.to(dtype=values.dtype))
