"""Traceable construction for the frozen legacy_train_es UEA protocol."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset.maskts import MaskTSDataset
from .dataset.load_from_tsfile_to_dataframe import load_from_tsfile_to_dataframe


PROTOCOL = "legacy_train_es"


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_legacy_dataset(prefix: Path, dataset: str, split: str, drop_ratio: float) -> MaskTSDataset:
    return MaskTSDataset(
        prefix=str(prefix),
        name=dataset,
        dataset=split,
        dataset_split_ratio=0.0,
        task="classification",
        mask_ratio=drop_ratio,
        mask_seed=None,
    )


def _raw_lengths(prefix: Path, dataset: str, split: str) -> list[list[int]]:
    raw, _ = load_from_tsfile_to_dataframe(str(prefix / dataset / f"{dataset}_{split.upper()}.ts"))
    return [[int(len(raw.loc[row, col])) for col in raw.columns] for row in range(len(raw))]


def mask_trace(dataset_obj: MaskTSDataset, prefix: Path, dataset: str, split: str, drop_ratio: float, seed: int) -> dict[str, Any]:
    raw_lengths = _raw_lengths(prefix, dataset, split)
    records = []
    for sample_id in range(dataset_obj.datasize):
        values = dataset_obj.feature.loc[sample_id]
        missing = values.isna().any(axis=1).to_numpy()
        lengths = raw_lengths[sample_id]
        start = int(dataset_obj._max_seq_len - max(lengths))
        original_indices = np.arange(start, dataset_obj._max_seq_len, dtype=np.int64)
        removed_indices = [int(i) for i in original_indices if missing[i]]
        kept_indices = [int(i) for i in original_indices if not missing[i]]
        records.append({
            "protocol": PROTOCOL,
            "dataset": dataset,
            "split": split,
            "drop_ratio": format(float(drop_ratio), ".17g"),
            "seed": int(seed),
            "sample_identity": int(sample_id),
            "raw_channel_lengths": lengths,
            "original_length": int(max(lengths)),
            "global_max_original_length": int(dataset_obj._max_seq_len),
            "legacy_axis_length": int(len(values)),
            # These are positions on the left-padded legacy axis.  Keeping both
            # lists makes the trace unambiguous even when a future dataset has
            # pre-existing missing values.
            "legacy_axis_removed_indices": removed_indices,
            "legacy_axis_kept_indices": kept_indices,
            "legacy_axis_padding_indices": [
                int(i) for i in range(start) if missing[i]
            ],
        })
    return {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "split": split,
        "drop_ratio": format(float(drop_ratio), ".17g"),
        "seed": int(seed),
        "record_count": len(records),
        "records": records,
        "hash": sha256_json(records),
    }


def trace_paths(preflight_dir: Path) -> tuple[Path, Path, Path]:
    return preflight_dir / "trace.json", preflight_dir / "rng_before_dataset.pt", preflight_dir / "rng_after_dataset.pt"


def create_preflight(prefix: Path, dataset: str, drop_ratio: float, seed: int, preflight_dir: Path, suite_hash: str) -> dict[str, Any]:
    """Construct legacy train/test data once and save canonical traces and RNG states."""
    preflight_dir.mkdir(parents=True, exist_ok=True)
    caller_state = capture_rng_state()
    set_all_seeds(seed)
    before = capture_rng_state()
    trainset = build_legacy_dataset(prefix, dataset, "train", drop_ratio)
    testset = build_legacy_dataset(prefix, dataset, "test", drop_ratio)
    train_trace = mask_trace(trainset, prefix, dataset, "train", drop_ratio, seed)
    test_trace = mask_trace(testset, prefix, dataset, "test", drop_ratio, seed)
    after = capture_rng_state()
    trace = {
        "protocol": PROTOCOL,
        "suite_hash": suite_hash,
        "dataset": dataset,
        "drop_ratio": format(float(drop_ratio), ".17g"),
        "seed": int(seed),
        "train": train_trace,
        "test": test_trace,
        "combined_hash": sha256_json({"train": train_trace["hash"], "test": test_trace["hash"]}),
        "rng_snapshot": {
            "python": True,
            "numpy": True,
            "torch_cpu": True,
            "torch_cuda": bool(before["torch_cuda"]),
        },
    }
    trace_path, before_path, after_path = trace_paths(preflight_dir)
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(before, before_path)
    torch.save(after, after_path)
    # Preflight is an audit-only construction.  It must not perturb the RNG
    # stream of the process that requested it.
    restore_rng_state(caller_state)
    return trace


def reconstruct_and_verify(prefix: Path, dataset: str, drop_ratio: float, seed: int, preflight_dir: Path, suite_hash: str):
    trace_path, before_path, _ = trace_paths(preflight_dir)
    expected = json.loads(trace_path.read_text(encoding="utf-8"))
    if expected["suite_hash"] != suite_hash:
        raise RuntimeError("Preflight suite hash does not match current source/configuration.")
    if not expected.get("rng_snapshot", {}).get("python") or not expected.get("rng_snapshot", {}).get("numpy"):
        raise RuntimeError("Preflight did not capture the required Python/NumPy RNG states.")
    restore_rng_state(torch.load(before_path, map_location="cpu", weights_only=False))
    trainset = build_legacy_dataset(prefix, dataset, "train", drop_ratio)
    testset = build_legacy_dataset(prefix, dataset, "test", drop_ratio)
    train_trace = mask_trace(trainset, prefix, dataset, "train", drop_ratio, seed)
    test_trace = mask_trace(testset, prefix, dataset, "test", drop_ratio, seed)
    actual = sha256_json({"train": train_trace["hash"], "test": test_trace["hash"]})
    if train_trace["hash"] != expected["train"]["hash"] or test_trace["hash"] != expected["test"]["hash"]:
        raise RuntimeError("Legacy mask trace mismatch before model construction.")
    if actual != expected["combined_hash"]:
        raise RuntimeError("Combined legacy mask hash mismatch before model construction.")
    actual_trace = {
        "train": train_trace,
        "test": test_trace,
        "combined_hash": actual,
    }
    return trainset, testset, expected, actual_trace


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
