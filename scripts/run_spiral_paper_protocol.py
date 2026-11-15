#!/usr/bin/env python
"""Run the paper-aligned Spiral robustness protocol from a manifest.

The defaults follow ContiFormer Appendix C.1.2 / Table 7: 300 trajectories,
200/100 train/test split, 150 grid points, 30 observations from the first
half, observation noise 0.1, 5000 optimization steps, and batch size 64.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_SEEDS = [13, 27, 42, 100, 2024]
DEFAULT_ALPHAS = [0.02, 0.2, 2.0]
DEFAULT_MODELS = ["transformer", "latent_ode", "contiformer", "hippogp"]
OPERATOR_PROFILES = {
    "operator_order_i",
    "operator_order_ii",
    "clean_fullgrid",
    "operator_order_all",
    "operator_order_pilot",
}
OPERATOR_TOPOLOGIES = ["parallel", "mha_only_matched", "mha_to_cm", "cm_to_mha"]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "transformer": {"model_name": "Transformer"},
    "latent_ode": {"model_name": "Neural_ODE"},
    "contiformer": {"model_name": "Contiformer"},
    "hippogp": {
        "model_name": "HiPPOGP_Analytic",
        "hippo_memory_normalization": "layernorm",
        "hippo_local_residual": True,
        "hippo_local_residual_type": "self_attn",
        "hippo_uncertainty_gate": False,
        "hippo_mixture_type": "default",
        "hippo_gp_rope": "none",
    },
}

METRIC_KEYS = [
    "interpolation_rmse",
    "interpolation_mae",
    "extrapolation_rmse",
    "extrapolation_mae",
]

PROTOCOL_SOURCE = "ContiFormer Appendix C.1.2 / Table 7"
PROTOCOL_NOTE = (
    "Appendix C.1.2 specifies 30 observations, while Section 4.1 states 50; "
    "this manifest follows the appendix and records nsample explicitly."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/spiral_table7_paper_protocol"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--profile",
        choices=["paper", *sorted(OPERATOR_PROFILES)],
        default="paper",
    )
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=DEFAULT_MODELS)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--niters", type=int, default=5000)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--nspiral", type=int, default=300)
    parser.add_argument("--ntrain", type=int, default=200)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--ntotal", type=int, default=150)
    parser.add_argument("--nsample", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--log-step", type=int, default=500)
    parser.add_argument("--visualize", type=str, default="0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def alpha_slug(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace(".", "p")


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def git_state(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def build_manifest(args: argparse.Namespace, out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_commit, source_dirty = git_state(args.root.resolve())
    for alpha in args.alphas:
        for seed in args.seeds:
            for model in args.models:
                run_dir = out_root / alpha_slug(alpha) / model / f"seed_{seed}"
                rows.append(
                    {
                        "task_index": len(rows),
                        "model": model,
                        "seed": seed,
                        "alpha": alpha,
                        "noise_std": args.noise_std,
                        "niters": args.niters,
                        "nspiral": args.nspiral,
                        "ntrain": args.ntrain,
                        "ntest": args.ntest,
                        "ntotal": args.ntotal,
                        "nsample": args.nsample,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "log_step": args.log_step,
                        "visualize": args.visualize,
                        "run_dir": str(run_dir),
                        "protocol_source": PROTOCOL_SOURCE,
                        "protocol_note": PROTOCOL_NOTE,
                        "source_commit": source_commit,
                        "source_dirty": source_dirty,
                        **MODEL_CONFIGS[model],
                    }
                )
    return rows


def _operator_protocols(profile: str) -> list[dict[str, Any]]:
    protocol_i = {
        "protocol": "protocol_i_benchmark",
        "model_protocol": "full_grid_linear",
        "selection_mode": "test_mae",
        "train_size": 200,
        "validation_size": 0,
        "test_size": 100,
        "topologies": OPERATOR_TOPOLOGIES,
    }
    protocol_ii = {
        "protocol": "protocol_ii_strict",
        "model_protocol": "strict_cm_readout",
        "selection_mode": "validation_mae",
        "train_size": 180,
        "validation_size": 20,
        "test_size": 100,
        "topologies": OPERATOR_TOPOLOGIES,
    }
    clean = {
        "protocol": "protocol_i_clean_selection",
        "model_protocol": "full_grid_linear",
        "selection_mode": "validation_mae",
        "train_size": 180,
        "validation_size": 20,
        "test_size": 100,
        "topologies": ["parallel"],
    }
    if profile == "operator_order_i":
        return [protocol_i]
    if profile == "operator_order_ii":
        return [protocol_ii]
    if profile == "clean_fullgrid":
        return [clean]
    return [protocol_i, protocol_ii, clean]


def build_operator_manifest(
    args: argparse.Namespace, out_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_commit, source_dirty = git_state(args.root.resolve())
    for protocol in _operator_protocols(args.profile):
        for nsample in (30, 50):
            for alpha in args.alphas:
                for seed in args.seeds:
                    for topology in protocol["topologies"]:
                        if args.profile == "operator_order_pilot" and not (
                            nsample == 30 and alpha == 0.02 and seed == 27
                        ):
                            continue
                        run_dir = (
                            out_root
                            / protocol["protocol"]
                            / f"nobs_{nsample}"
                            / alpha_slug(alpha)
                            / topology
                            / f"seed_{seed}"
                        )
                        rows.append(
                            {
                                "task_index": len(rows),
                                "task_id": f"spiral_order_{len(rows):04d}",
                                "experiment": "spiral_operator_order",
                                "protocol": protocol["protocol"],
                                "model_protocol": protocol["model_protocol"],
                                "selection_mode": protocol["selection_mode"],
                                "train_size": protocol["train_size"],
                                "validation_size": protocol["validation_size"],
                                "test_size": protocol["test_size"],
                                "topology": topology,
                                "model": f"v3_{topology}",
                                "seed": seed,
                                "alpha": alpha,
                                "noise_std": args.noise_std,
                                "niters": 2
                                if args.profile == "operator_order_pilot"
                                else args.niters,
                                "eval_every": 1,
                                "nspiral": args.nspiral,
                                "ntotal": args.ntotal,
                                "nsample": nsample,
                                "batch_size": args.batch_size,
                                "lr": args.lr,
                                "dropout": 0.0,
                                "memory_size": 16,
                                "local_residual_init": 0.1,
                                "uncertainty_gate": False,
                                "run_dir": str(run_dir),
                                "protocol_source": PROTOCOL_SOURCE,
                                "protocol_note": (
                                    "Operator-order audit with both Appendix Nobs=30 and "
                                    "Section 4.1 Nobs=50. Protocol I retains released-code "
                                    "test selection; Protocol II and clean control use 180/20/100."
                                ),
                                "source_commit": source_commit,
                                "source_dirty": source_dirty,
                            }
                        )
    return rows


def bool_arg(value: bool) -> str:
    return "True" if value else "False"


def build_command(root: Path, row: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(root / "SeqML" / "ContiFormer" / "spiral.py"),
        "--model_name",
        str(row["model_name"]),
        "--seed",
        str(row["seed"]),
        "--noise_a",
        str(row["alpha"]),
        "--noise_std",
        str(row["noise_std"]),
        "--niters",
        str(row["niters"]),
        "--nspiral",
        str(row["nspiral"]),
        "--ntrain",
        str(row["ntrain"]),
        "--ntest",
        str(row["ntest"]),
        "--ntotal",
        str(row["ntotal"]),
        "--nsample",
        str(row["nsample"]),
        "--batch_size",
        str(row["batch_size"]),
        "--lr",
        str(row["lr"]),
        "--log_step",
        str(row["log_step"]),
        "--visualize",
        str(row["visualize"]),
        "--adjoint",
        "True",
        "--cc",
        "2",
        "--train_dir",
        str(row["run_dir"]),
    ]
    for key in [
        "hippo_memory_normalization",
        "hippo_local_residual_type",
        "hippo_mixture_type",
        "hippo_gp_rope",
    ]:
        if key in row:
            command.extend([f"--{key}", str(row[key])])
    for key in ["hippo_local_residual", "hippo_uncertainty_gate"]:
        if key in row:
            command.extend([f"--{key}", bool_arg(bool(row[key]))])
    return command


def load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def metrics_from_pred(path: Path) -> dict[str, float]:
    import torch

    payload = load_checkpoint(path)
    pred = payload["pred"].detach().cpu()
    target = payload["target"].detach().cpu()
    split = pred.shape[1] // 2
    slices = {
        "interpolation": (pred[:, :split, :], target[:, :split, :]),
        "extrapolation": (pred[:, split - 1 :, :], target[:, split - 1 :, :]),
    }
    result: dict[str, float] = {}
    for name, (prediction, truth) in slices.items():
        difference = prediction - truth
        result[f"{name}_mae"] = torch.abs(difference).sum(dim=-1).mean().item()
        result[f"{name}_rmse"] = torch.sqrt(difference.square().sum(dim=-1).mean()).item()
    return result


def best_log_metrics(path: Path) -> dict[str, float | int]:
    pattern = re.compile(r"Iter: (\d+), MAE: ([0-9.]+), RMSE: ([0-9.]+)")
    best: dict[str, float | int] = {}
    if not path.exists():
        return best
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        iteration, mae, rmse = int(match.group(1)), float(match.group(2)), float(match.group(3))
        if not best or mae < float(best["best_mae"]):
            best = {"best_iter": iteration, "best_mae": mae, "best_rmse": rmse}
    return best


def summarize(manifest: list[dict[str, Any]], out_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for task in manifest:
        run_dir = Path(task["run_dir"])
        pred_path = run_dir / "pred.pkl"
        if not pred_path.exists():
            missing.append(task)
            continue
        row = {
            "model": task["model"],
            "seed": task["seed"],
            "alpha": task["alpha"],
            "noise_std": task["noise_std"],
            "niters": task["niters"],
            "nsample": task["nsample"],
            "batch_size": task["batch_size"],
            "run_dir": str(run_dir),
        }
        row.update(metrics_from_pred(pred_path))
        row.update(best_log_metrics(run_dir / "log.log"))
        rows.append(row)

    fieldnames = [
        "model", "seed", "alpha", "noise_std", "niters", "nsample", "batch_size",
        *METRIC_KEYS, "best_iter", "best_mae", "best_rmse", "run_dir",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "paper_protocol_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    aggregate: list[dict[str, Any]] = []
    groups = sorted({(float(row["alpha"]), str(row["model"])) for row in rows})
    for alpha, model in groups:
        selected = [row for row in rows if float(row["alpha"]) == alpha and row["model"] == model]
        summary_row: dict[str, Any] = {"alpha": alpha, "model": model, "runs": len(selected)}
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in selected]
            summary_row[f"{key}_mean"] = mean(values)
            summary_row[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate.append(summary_row)

    aggregate_fieldnames = ["alpha", "model", "runs"]
    for key in METRIC_KEYS:
        aggregate_fieldnames.extend([f"{key}_mean", f"{key}_std"])
    with (out_root / "paper_protocol_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fieldnames)
        writer.writeheader()
        writer.writerows(aggregate)

    (out_root / "paper_protocol_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_root / "paper_protocol_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (out_root / "paper_protocol_missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    print(f"Completed {len(rows)}/{len(manifest)} tasks; missing {len(missing)}")
    print(f"Wrote summaries to {out_root}")


def summarize_operator_order(manifest: list[dict[str, Any]], out_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for task in manifest:
        metrics_path = Path(task["run_dir"]) / "metrics.json"
        if not metrics_path.exists():
            missing.append(task)
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "protocol": task["protocol"],
            "topology": task["topology"],
            "nsample": task["nsample"],
            "alpha": task["alpha"],
            "seed": task["seed"],
            "selection_mode": task["selection_mode"],
            "best_iteration": payload["best_iteration"],
            "best_selection_mae": payload["best_selection_mae"],
            "parameter_count": payload["parameter_count"],
            "total_seconds": payload["runtime"]["total_seconds"],
            "run_dir": task["run_dir"],
            **payload["metrics"],
        }
        rows.append(row)

    metric_names = [*METRIC_KEYS, "overall_mae", "overall_rmse"]
    fields = [
        "protocol", "topology", "nsample", "alpha", "seed", "selection_mode",
        *metric_names, "best_iteration", "best_selection_mae", "parameter_count",
        "total_seconds", "run_dir",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "operator_order_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    aggregate: list[dict[str, Any]] = []
    groups = sorted(
        {
            (row["protocol"], row["topology"], int(row["nsample"]), float(row["alpha"]))
            for row in rows
        }
    )
    for protocol, topology, nsample, alpha in groups:
        selected = [
            row
            for row in rows
            if row["protocol"] == protocol
            and row["topology"] == topology
            and int(row["nsample"]) == nsample
            and float(row["alpha"]) == alpha
        ]
        summary_row: dict[str, Any] = {
            "protocol": protocol,
            "topology": topology,
            "nsample": nsample,
            "alpha": alpha,
            "runs": len(selected),
        }
        for key in metric_names:
            values = [float(row[key]) for row in selected]
            summary_row[f"{key}_mean"] = mean(values)
            summary_row[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate.append(summary_row)

    aggregate_fields = ["protocol", "topology", "nsample", "alpha", "runs"]
    for key in metric_names:
        aggregate_fields.extend([f"{key}_mean", f"{key}_std"])
    with (out_root / "operator_order_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate)
    (out_root / "operator_order_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (out_root / "operator_order_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (out_root / "operator_order_missing.json").write_text(
        json.dumps(missing, indent=2), encoding="utf-8"
    )
    print(f"Completed {len(rows)}/{len(manifest)} tasks; missing {len(missing)}")
    print(f"Wrote operator-order summaries to {out_root}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.self_test:
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "PhysioPro"))
        from experiments.spiral_operator_order import architecture_self_test

        print(json.dumps(architecture_self_test(), indent=2, sort_keys=True))
        if not args.write_manifest and args.task_index is None and not args.summarize:
            return
    if args.profile in OPERATOR_PROFILES and args.out_root == Path(
        "outputs/spiral_table7_paper_protocol"
    ):
        args.out_root = Path(
            "outputs/spiral_operator_order_pilot"
            if args.profile == "operator_order_pilot"
            else "outputs/spiral_operator_order"
        )
    out_root = resolve_path(root, args.out_root).resolve()
    manifest_path = args.manifest or out_root / "manifest.jsonl"
    manifest_path = resolve_path(root, manifest_path).resolve()

    if args.write_manifest:
        rows = (
            build_operator_manifest(args, out_root)
            if args.profile in OPERATOR_PROFILES
            else build_manifest(args, out_root)
        )
        write_jsonl(manifest_path, rows)
        print(f"Wrote {len(rows)} tasks to {manifest_path}")
        if args.task_index is None and not args.summarize:
            return

    if not manifest_path.exists():
        raise FileNotFoundError(f"Generate the manifest first: {manifest_path}")
    manifest = read_jsonl(manifest_path)

    if args.summarize:
        if manifest and manifest[0].get("experiment") == "spiral_operator_order":
            summary_root = manifest_path.parent if args.manifest is not None else out_root
            summarize_operator_order(manifest, summary_root)
        else:
            summarize(manifest, out_root)
        if args.task_index is None:
            return

    task_index = args.task_index
    if task_index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_index is None:
        print(f"Manifest contains {len(manifest)} tasks; pass --task-index to run one.")
        return
    if task_index < 0 or task_index >= len(manifest):
        raise IndexError(f"Task index {task_index} is outside [0, {len(manifest) - 1}]")

    row = manifest[task_index]
    run_dir = Path(row["run_dir"])
    if row.get("experiment") == "spiral_operator_order":
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "PhysioPro"))
        from experiments.mechanism_audit_common import (
            completed_config_matches,
            mark_success,
            seed_everything,
            write_run_metadata,
        )

        if args.skip_existing and completed_config_matches(run_dir, row):
            print(f"Skipping completed task {task_index}: {run_dir}")
            return
        if args.dry_run:
            print(json.dumps(row, indent=2, sort_keys=True))
            return
        seed_everything(int(row["seed"]))
        write_run_metadata(run_dir, row, root)
        from experiments.spiral_operator_order import run_operator_order_experiment

        run_operator_order_experiment(row, run_dir)
        mark_success(run_dir)
        print(f"Completed operator-order task {task_index}: {run_dir}")
        return
    pred_path = run_dir / "pred.pkl"
    if args.skip_existing and pred_path.exists():
        print(f"Skipping completed task {task_index}: {pred_path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "paper_protocol_config.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    command = build_command(root, row)
    print("+ " + " ".join(command), flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "PhysioPro"), env.get("PYTHONPATH", "")]
    )
    subprocess.run(command, cwd=str(root), env=env, check=True)


if __name__ == "__main__":
    main()
