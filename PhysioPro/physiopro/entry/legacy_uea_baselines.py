"""Single-task entry point for traceable ``legacy_train_es`` UEA baselines."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml
from utilsd import use_cuda

from physiopro.legacy_uea import (
    PROTOCOL,
    git_commit,
    reconstruct_and_verify,
    set_all_seeds,
    sha256_file,
)
from physiopro.model.legacy_uea import LegacyBaselineMaskTS
from physiopro.network import NETWORKS
import physiopro.network.legacy_uea_baselines  # Registers isolated networks.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--drop-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--suite-hash", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resource-preflight", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-gpu-substring", default=None)
    return parser.parse_args()


def gpu_metadata() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"gpu_model": "none", "cuda_available": False}
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = "unknown"
    return {
        "gpu_model": torch.cuda.get_device_name(0),
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_total_memory_mb": float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)),
        "driver_version": driver,
        "allocated_memory_mb": float(torch.cuda.memory_allocated() / (1024 ** 2)),
        "reserved_memory_mb": float(torch.cuda.memory_reserved() / (1024 ** 2)),
        "slurm_gres": __import__("os").environ.get("SLURM_JOB_GRES", __import__("os").environ.get("SLURM_STEP_GPUS", "unknown")),
    }


def resource_preflight(model: LegacyBaselineMaskTS, trainset) -> dict[str, object]:
    """Run one real batch-64 forward/backward without changing the protocol."""
    from torch.utils.data import DataLoader
    from physiopro.common.utils import to_torch

    trainset.load()
    loader = DataLoader(trainset, batch_size=model.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    data, label = next(iter(loader))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    data, label = to_torch(data, device="cuda"), to_torch(label, device="cuda")
    prediction = model(data)
    loss = model.loss_fn(label.squeeze(-1), prediction.squeeze(-1))
    model.optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    trainset.freeup()
    if not torch.isfinite(loss):
        raise RuntimeError("Resource preflight produced a non-finite loss.")
    return {
        "batch_size": model.batch_size,
        "loss": float(loss.detach().cpu()),
        "elapsed_sec": elapsed,
        "peak_allocated_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "peak_reserved_memory_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["id"] != args.model:
        raise ValueError("Config ID does not match --model.")
    use_cuda(torch.cuda.is_available())
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("This pilot worker requires an allocated CUDA GPU.")
    if args.expected_gpu_substring and (
        not torch.cuda.is_available()
        or args.expected_gpu_substring.lower() not in torch.cuda.get_device_name(0).lower()
    ):
        raise RuntimeError(
            f"Expected GPU containing {args.expected_gpu_substring!r}, got {gpu_metadata()['gpu_model']!r}."
        )

    # Establish a deterministic CUDA fallback before restoring the full
    # preflight snapshot.  A valid GPU preflight then replaces it entirely.
    set_all_seeds(args.seed)
    # This occurs before model construction and before any DataLoader iteration.
    trainset, testset, trace, actual_trace = reconstruct_and_verify(
        args.prefix, args.dataset, args.drop_ratio, args.seed, args.preflight_dir, args.suite_hash
    )
    if args.require_cuda and not trace.get("rng_snapshot", {}).get("torch_cuda"):
        raise RuntimeError("GPU task requires a preflight trace with a CUDA RNG snapshot.")
    if args.dry_run:
        print(json.dumps({"trace_hash": trace["combined_hash"], "status": "preflight-ok"}, indent=2))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    network = NETWORKS.get(config["network"]["type"])(
        input_size=trainset.num_variables,
        max_length=trainset.max_seq_len,
        **{k: v for k, v in config["network"].items() if k != "type"},
    )
    model_config = dict(config["model"])
    s5_options = model_config.pop("s5_options", {})
    model = LegacyBaselineMaskTS(
        network=network,
        output_dir=args.out_dir,
        checkpoint_dir=checkpoint_dir,
        out_size=trainset.num_classes,
        epoch_metrics_path=args.out_dir / "epoch_metrics.jsonl",
        **s5_options,
        **model_config,
    )
    if args.resource_preflight:
        result = resource_preflight(model, trainset)
        result.update({
            "protocol": PROTOCOL,
            "dataset": args.dataset,
            "drop_ratio": args.drop_ratio,
            "seed": args.seed,
            "internal_model_id": args.model,
            "suite_hash": args.suite_hash,
            "mask_hash": actual_trace["combined_hash"],
            "hardware": gpu_metadata(),
            "efficiency_scope": "diagnostic_only",
        })
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "resource_preflight.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    started = time.time()
    model.fit(trainset, None, testset)
    model.predict(trainset, "train")
    model.predict(testset, "test")
    elapsed = time.time() - started
    meta = {
        "protocol": PROTOCOL,
        "internal_model_id": args.model,
        "paper_display_label": config["paper_label"],
        "dataset": args.dataset,
        "drop_ratio": args.drop_ratio,
        "seed": args.seed,
        "mask_hash_train": actual_trace["train"]["hash"],
        "mask_hash_test": actual_trace["test"]["hash"],
        "mask_hash": actual_trace["combined_hash"],
        "expected_mask_hash_train": trace["train"]["hash"],
        "expected_mask_hash_test": trace["test"]["hash"],
        "expected_mask_hash": trace["combined_hash"],
        "config_hash": sha256_file(config_path),
        "suite_hash": args.suite_hash,
        "git_commit": git_commit(root),
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_training_time_sec": elapsed,
        "framework_version": {"python": platform.python_version(), "torch": torch.__version__},
        "config_path": str(config_path),
        "efficiency_scope": "diagnostic_only",
        "hardware": gpu_metadata(),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
