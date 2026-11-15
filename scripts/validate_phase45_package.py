"""Validate the sanitized supplemental Phase 4.5 package."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "rebuttal" / "phase45_public"
REQUIRED = [
    PACKAGE / "results" / "final_ablation_paired.csv",
    PACKAGE / "results" / "final_ablation_summary.csv",
    PACKAGE / "results" / "final_rank_scaling.csv",
    PACKAGE / "results" / "final_query_scaling.csv",
    PACKAGE / "results" / "final_observation_scaling.csv",
    PACKAGE / "results" / "final_wallclock.csv",
    PACKAGE / "results" / "final_time_to_quality.csv",
    PACKAGE / "tables" / "table_ablation.tex",
    PACKAGE / "tables" / "table_rank_scaling.tex",
    PACKAGE / "tables" / "table_query_scaling.tex",
    PACKAGE / "tables" / "table_observation_scaling.tex",
    PACKAGE / "tables" / "table_wallclock.tex",
    PACKAGE / "tables" / "table_time_to_quality.tex",
]
REMOVED_VARIANTS = {"no_kernel", "mismatched_geometry", "simple_projection"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        raise SystemExit("Missing package files: " + ", ".join(missing))

    paired = rows(PACKAGE / "results" / "final_ablation_paired.csv")
    summary = rows(PACKAGE / "results" / "final_ablation_summary.csv")
    bad = sorted({row["variant"] for row in paired + summary} & REMOVED_VARIANTS)
    if bad:
        raise SystemExit("Removed ablation variants still present: " + ", ".join(bad))

    manifest = PACKAGE / "PACKAGE_MANIFEST.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                path = PACKAGE / item["path"]
                if not path.exists():
                    raise SystemExit(f"Manifest file missing: {item['path']}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != item["sha256"]:
                    raise SystemExit(f"Manifest hash mismatch: {item['path']}")

    print("Phase 4.5 package validation passed")
    print(f"ablation paired rows: {len(paired)}")
    print(f"ablation summary rows: {len(summary)}")
    print("removed variants absent: no_kernel, mismatched_geometry, simple_projection")


if __name__ == "__main__":
    main()
