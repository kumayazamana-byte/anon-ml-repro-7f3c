"""Run the paper UEA protocol with only the baseline and original HiPPO-FM arms.

This wrapper deliberately excludes later branch variants and historical result
archives. It supports dry-run command inspection before any training starts.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

CONFIGS = {
    "baseline_contiformer": "docs/configs/contiformer_uea_mask_classification.yml",
    "hippogp_analytic": "docs/configs/hippogp_contiformer_analytic_mask_classification.yml",
}
DEFAULT_DATASETS = ["Heartbeat"]
DEFAULT_DROPS = [0.3, 0.5, 0.7]
DEFAULT_SEEDS = [27, 42, 1024]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--drop-ratios", nargs="+", type=float, default=DEFAULT_DROPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/uea_paper_protocol"))
    parser.add_argument("--dataset-split-ratio", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    physiopro = root / "PhysioPro"
    out_root = args.out_root if args.out_root.is_absolute() else root / args.out_root
    env = os.environ.copy()
    env["PYTHONPATH"] = str(physiopro) + os.pathsep + env.get("PYTHONPATH", "")
    for dataset in args.datasets:
        for drop in args.drop_ratios:
            label = f"{drop:g}"
            for variant in args.variants:
                for seed in args.seeds:
                    output_dir = out_root / dataset / f"drop_{label}" / variant / f"seed_{seed}"
                    cmd = [
                        "python", "-m", "physiopro.entry.train", CONFIGS[variant],
                        "--data.mask_ratio", str(drop),
                        "--data.name", dataset,
                        "--data.dataset_split_ratio", str(args.dataset_split_ratio),
                        "--runtime.seed", str(seed),
                        "--runtime.output_dir", str(output_dir),
                    ]
                    print("+ " + " ".join(cmd))
                    if not args.dry_run:
                        subprocess.run(cmd, cwd=physiopro, env=env, check=True)


if __name__ == "__main__":
    main()
