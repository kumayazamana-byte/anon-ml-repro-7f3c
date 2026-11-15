#!/usr/bin/env python
"""Run missing Protocol-I Clean Spiral models from an auditable manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "spiral_operator_order"
DEFAULT_MODELS = [
    "mha_to_cm",
    "transformer",
    "neural_ode",
    "latent_ode",
    "neural_cde",
    "contiformer",
]
SEEDS = [13, 27, 42, 100, 2024]
ALPHAS = [0.02, 0.2, 2.0]
NSAMPLES = [30, 50]
METRICS = [
    "interpolation_rmse",
    "interpolation_mae",
    "extrapolation_rmse",
    "extrapolation_mae",
    "overall_mae",
    "overall_rmse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--nsamples", nargs="+", type=int, choices=NSAMPLES, default=NSAMPLES)
    parser.add_argument("--niters", type=int, default=5000)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "PhysioPro",
                "SeqML",
                "experiments",
                "scripts",
                "cluster",
            ],
            cwd=root,
            text=True,
        ).strip()
    )
    return commit, dirty


def alpha_slug(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace(".", "p")


def build_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_commit, source_dirty = git_state(args.root)
    rows: list[dict[str, Any]] = []
    for model in args.models:
        for nsample in args.nsamples:
            for alpha in args.alphas:
                for seed in args.seeds:
                    run_dir = (
                        args.out_root
                        / "protocol_i_clean_independent_validation"
                        / f"nobs_{nsample}"
                        / alpha_slug(alpha)
                        / model
                        / f"seed_{seed}"
                    )
                    rows.append(
                        {
                            "task_index": len(rows),
                            "task_id": f"spiral_clean_{model}_n{nsample}_a{alpha:g}_s{seed}",
                            "experiment": "spiral_protocol_i_clean",
                            "protocol": "protocol_i_clean_independent_validation",
                            "model_protocol": (
                                "full_grid_linear"
                                if model in {"parallel", "mha_to_cm"}
                                else "released_baseline_readout"
                                if model in {"transformer", "latent_ode", "contiformer"}
                                else "matched_reimplementation"
                            ),
                            "implementation_source": (
                                "experiments.spiral_operator_order"
                                if model in {"parallel", "mha_to_cm"}
                                else "SeqML/ContiFormer/spiral.py"
                                if model in {"transformer", "latent_ode", "contiformer"}
                                else "experiments.spiral_protocol_i_clean_baselines"
                            ),
                            "selection_mode": "independent_validation_mae",
                            "train_size": 200,
                            "validation_size": 50,
                            "test_size": 100,
                            "model": model,
                            "topology": model if model in {"parallel", "mha_to_cm"} else None,
                            "seed": seed,
                            "validation_seed": seed + 1_000_000,
                            "alpha": alpha,
                            "noise_std": 0.1,
                            "niters": args.niters,
                            "eval_every": 1,
                            "nspiral": 300,
                            "ntotal": 150,
                            "nsample": nsample,
                            "batch_size": 64,
                            "lr": 0.01,
                            "dropout": 0.0,
                            "memory_size": 16,
                            "run_dir": str(run_dir),
                            "source_commit": source_commit,
                            "source_dirty": source_dirty,
                        }
                    )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for task in rows:
        run_dir = Path(task["run_dir"])
        path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        try:
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved_config = None
        if not (run_dir / "_SUCCESS").exists() or saved_config != task or not path.exists():
            missing.append(task)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "protocol": task["protocol"],
                "model": task["model"],
                "nsample": task["nsample"],
                "alpha": task["alpha"],
                "seed": task["seed"],
                "selection_mode": task["selection_mode"],
                **payload["metrics"],
                "best_iteration": payload["best_iteration"],
                "best_selection_mae": payload["best_selection_mae"],
                "parameter_count": payload["parameter_count"],
                "total_seconds": payload["runtime"]["total_seconds"],
                "run_dir": task["run_dir"],
            }
        )
    output = manifest_path.parent
    stem = manifest_path.stem
    fields = [
        "protocol", "model", "nsample", "alpha", "seed", "selection_mode",
        *METRICS, "best_iteration", "best_selection_mae", "parameter_count",
        "total_seconds", "run_dir",
    ]
    with (output / f"{stem}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    aggregate: list[dict[str, Any]] = []
    groups = sorted({(row["model"], int(row["nsample"]), float(row["alpha"])) for row in records})
    for model, nsample, alpha in groups:
        selected = [
            row for row in records
            if row["model"] == model and int(row["nsample"]) == nsample and float(row["alpha"]) == alpha
        ]
        item: dict[str, Any] = {"model": model, "nsample": nsample, "alpha": alpha, "runs": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate.append(item)
    aggregate_fields = ["model", "nsample", "alpha", "runs"]
    for metric in METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    with (output / f"{stem}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate)
    (output / f"{stem}_missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    print(f"Completed {len(records)}/{len(rows)} tasks; missing {len(missing)}")


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.out_root = args.out_root.resolve()
    args.manifest = args.manifest.resolve()
    if args.write_manifest:
        rows = build_manifest(args)
        write_jsonl(args.manifest, rows)
        print(f"Wrote {len(rows)} tasks to {args.manifest}")
        if args.task_index is None and not args.summarize:
            return
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    rows = read_jsonl(args.manifest)
    if args.summarize:
        summarize(rows, args.manifest)
        if args.task_index is None:
            return
    task_index = args.task_index
    if task_index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_index is None:
        print(f"Manifest contains {len(rows)} tasks; pass --task-index to run one.")
        return
    task = rows[task_index]
    run_dir = Path(task["run_dir"])
    sys.path.insert(0, str(args.root))
    sys.path.insert(0, str(args.root / "PhysioPro"))
    from experiments.mechanism_audit_common import (
        completed_config_matches,
        mark_success,
        seed_everything,
        write_run_metadata,
    )

    if args.skip_existing and completed_config_matches(run_dir, task):
        print(f"Skipping completed task {task_index}: {run_dir}")
        return
    if args.dry_run:
        print(json.dumps(task, indent=2, sort_keys=True))
        return
    seed_everything(int(task["seed"]))
    success_path = run_dir / "_SUCCESS"
    if success_path.exists():
        success_path.unlink()
    write_run_metadata(run_dir, task, args.root)
    if task["model"] in {"parallel", "mha_to_cm"}:
        from experiments.spiral_operator_order import run_operator_order_experiment

        run_operator_order_experiment(task, run_dir)
    else:
        from experiments.spiral_protocol_i_clean_baselines import run_clean_baseline_experiment

        run_clean_baseline_experiment(task, run_dir)
    mark_success(run_dir)
    print(f"Completed Protocol-I Clean task {task_index}: {run_dir}")


if __name__ == "__main__":
    main()
