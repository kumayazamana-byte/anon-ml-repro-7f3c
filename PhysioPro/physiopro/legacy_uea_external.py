"""Utilities for the isolated external UEA baseline suite.

The external adapters use the same legacy ``MaskTS.merge`` construction as the
frozen TST/S5/MHA suite.  This module only pairs a newly generated preflight
trace with an existing trace; it does not construct datasets or change the
legacy masking process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def trace_relative_path(dataset: str, drop_ratio: float | str, seed: int) -> Path:
    drop = format(float(drop_ratio), ".1f")
    return Path(dataset) / f"drop_{drop}" / f"seed_{int(seed)}" / "trace.json"


def candidate_trace_paths(root: Path, dataset: str, drop_ratio: float | str, seed: int) -> list[Path]:
    """Return likely trace locations for either a preflight root or repo root."""
    relative = trace_relative_path(dataset, drop_ratio, seed)
    root = root.resolve()
    candidates = [root / relative, root / "preflight" / relative]
    return list(dict.fromkeys(candidates))


def default_reference_roots(repo_root: Path) -> list[Path]:
    """Locate frozen traces produced by the previous legacy baseline suite."""
    return [
        repo_root / "results" / "uea_legacy_train_es_3090ti_partial_20260819" / "preflight",
        repo_root / "results" / "uea_legacy_s5_4090_repair_20260820" / "preflight",
        repo_root / "outputs" / "UEA_legacy_train_es_baselines" / "preflight",
    ]


def find_reference_trace(
    roots: Iterable[Path], dataset: str, drop_ratio: float | str, seed: int
) -> Path | None:
    for root in roots:
        for candidate in candidate_trace_paths(Path(root), dataset, drop_ratio, seed):
            if candidate.is_file():
                return candidate
    return None


def _load_trace(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Trace is not a JSON object: {path}")
    return value


def compare_reference_trace(
    current_trace: dict[str, Any],
    reference_path: Path | None,
    *,
    dataset: str,
    drop_ratio: float | str,
    seed: int,
) -> dict[str, Any]:
    """Compare mask identities while deliberately ignoring suite hashes.

    The source/configuration hash changes for the external suite, but the
    train/test mask identities must remain equal to the frozen reference.
    """
    result: dict[str, Any] = {
        "reference_available": reference_path is not None,
        "reference_match": False,
        "reference_path": str(reference_path) if reference_path is not None else None,
        "dataset": dataset,
        "drop_ratio": format(float(drop_ratio), ".1f"),
        "seed": int(seed),
        "reason": "reference trace unavailable",
    }
    if reference_path is None:
        return result
    try:
        reference = _load_trace(reference_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["reason"] = f"invalid reference trace: {exc}"
        return result
    identity_fields = ("dataset", "drop_ratio", "seed", "combined_hash")
    if any(current_trace.get(field) != reference.get(field) for field in identity_fields):
        result["reason"] = "dataset/drop/seed/combined hash mismatch"
        result["reference_combined_hash"] = reference.get("combined_hash")
        return result
    if not all(
        isinstance(current_trace.get(split), dict)
        and isinstance(reference.get(split), dict)
        and current_trace[split].get("hash") == reference[split].get("hash")
        for split in ("train", "test")
    ):
        result["reason"] = "train/test mask hash mismatch"
        result["reference_combined_hash"] = reference.get("combined_hash")
        return result
    result.update({
        "reference_match": True,
        "reference_combined_hash": reference.get("combined_hash"),
        "reason": "train/test mask hashes match",
    })
    return result


def write_reference_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
