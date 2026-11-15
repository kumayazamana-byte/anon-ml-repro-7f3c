"""LSST-only finite-gated entry for the grouped-length MambaSL repair."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
import yaml
from utilsd import use_cuda

from physiopro.entry.legacy_uea_mha_to_cm import (
    gpu_metadata,
    reference_status,
    resource_preflight,
)
from physiopro.entry.legacy_uea_mambasl import validate_upstream_config
from physiopro.legacy_uea import (
    PROTOCOL,
    git_commit,
    reconstruct_and_verify,
    set_all_seeds,
    sha256_file,
)
from physiopro.model.legacy_uea_mambasl import LegacyMambaSLMaskTS
from physiopro.network.legacy_uea_mambasl_grouped import GroupedLengthMambaSLAdapter


SUITE = "uea_legacy_train_es_mambasl_lsst_grouped_repair"


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
    parser.add_argument("--reference-trace-root", action="append", default=[])
    parser.add_argument("--allow-unpaired-reference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resource-preflight", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-gpu-substring", default=None)
    return parser.parse_args()


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def main() -> None:
    args = parse_args()
    if args.dataset != "LSST" or args.drop_ratio not in {0.3, 0.5, 0.7}:
        raise ValueError("This repair entry accepts only the nine formal LSST cells.")
    root = args.root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("id") != args.model or config.get("suite") != SUITE:
        raise ValueError("Config identity or suite does not match this worker.")
    validate_upstream_config(config)

    use_cuda(torch.cuda.is_available())
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("This worker requires an allocated CUDA GPU.")
    if args.expected_gpu_substring and (
        not torch.cuda.is_available()
        or args.expected_gpu_substring.lower() not in torch.cuda.get_device_name(0).lower()
    ):
        raise RuntimeError(
            f"Expected GPU containing {args.expected_gpu_substring!r}, "
            f"got {gpu_metadata()['gpu_model']!r}."
        )

    set_all_seeds(args.seed)
    trainset, testset, trace, actual_trace = reconstruct_and_verify(
        args.prefix, args.dataset, args.drop_ratio, args.seed,
        args.preflight_dir, args.suite_hash,
    )
    reference = reference_status(args, trace, root)
    if args.require_cuda and not trace.get("rng_snapshot", {}).get("torch_cuda"):
        raise RuntimeError("GPU task requires a preflight trace with a CUDA RNG snapshot.")
    if args.dry_run:
        print(json.dumps({
            "trace_hash": trace["combined_hash"],
            "reference_trace_match": reference["reference_match"],
            "status": "preflight-ok",
        }, indent=2, sort_keys=True))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    network_config = config["network"]
    network = GroupedLengthMambaSLAdapter(
        input_size=trainset.num_variables,
        max_length=trainset.max_seq_len,
        num_classes=trainset.num_classes,
        d_model=int(network_config["d_model"]),
        d_state=int(network_config["d_state"]),
        d_conv=int(network_config["d_conv"]),
        expand=int(network_config["expand"]),
        n_heads=int(network_config["n_heads"]),
        dropout=float(network_config["dropout"]),
        tv_dt=bool(network_config["tv_dt"]),
        tv_B=bool(network_config["tv_B"]),
        tv_C=bool(network_config["tv_C"]),
        use_D=bool(network_config["use_D"]),
    )
    model = LegacyMambaSLMaskTS(
        network=network,
        output_dir=args.out_dir,
        checkpoint_dir=checkpoint_dir,
        out_size=trainset.num_classes,
        epoch_metrics_path=args.out_dir / "epoch_metrics.jsonl",
        **config["model"],
    )
    if args.resource_preflight:
        result = resource_preflight(model, trainset)
        result.update({
            "protocol": PROTOCOL,
            "suite": SUITE,
            "dataset": args.dataset,
            "drop_ratio": args.drop_ratio,
            "seed": args.seed,
            "internal_model_id": args.model,
            "suite_hash": args.suite_hash,
            "mask_hash": actual_trace["combined_hash"],
            "reference_trace_match": reference["reference_match"],
            "hardware": gpu_metadata(),
            "adapter_execution": network.execution_mode,
            "upstream": network.upstream_provenance,
            "efficiency_scope": "diagnostic_only",
        })
        (args.out_dir / "resource_preflight.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    started = time.time()
    model.fit(trainset, None, testset)
    model.predict(trainset, "train")
    model.predict(testset, "test")
    result = json.loads((checkpoint_dir / "res.json").read_text(encoding="utf-8"))
    if not finite({"train": result.get("train"), "test": result.get("test")}):
        failure = {
            "status": "nonfinite",
            "dataset": args.dataset,
            "drop_ratio": args.drop_ratio,
            "seed": args.seed,
            "result": result,
        }
        (args.out_dir / "finite_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise RuntimeError("MambaSL repair completed with non-finite train/test metrics.")

    metadata = {
        "protocol": PROTOCOL,
        "suite": SUITE,
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
        "reference_trace_match": reference["reference_match"],
        "reference_trace_path": reference.get("reference_path"),
        "config_hash": sha256_file(config_path),
        "suite_hash": args.suite_hash,
        "git_commit": git_commit(root),
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_training_time_sec": time.time() - started,
        "framework_version": {"python": platform.python_version(), "torch": torch.__version__},
        "config_path": str(config_path),
        "upstream": network.upstream_provenance,
        "num_kernels": network.num_kernels,
        "adapter_execution": network.execution_mode,
        "adaptation": (
            "Each sample keeps exactly the legacy MaskTS retained rows. Samples are grouped "
            "by valid length and right padding is removed before the unchanged pinned upstream "
            "MambaSL model is called; timestamps remain unused."
        ),
        "efficiency_scope": "diagnostic_only",
        "hardware": gpu_metadata(),
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
