# Anonymous reproducibility package

This repository contains the model implementation and the training entry points needed to reproduce the paper's protocol-defined Spiral and UEA experiments. It is an anonymous review export: it contains no author names, email addresses, personal paths, checkpoints, raw datasets, result archives, or prior rebuttal evidence.

## Environment

Use Python 3.10 and a PyTorch build compatible with the target GPU. Install the local package and the small dependency set:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -e PhysioPro
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Smoke checks

```bash
bash reproduce.sh smoke
python scripts/run_spiral_paper_protocol.py --help
python scripts/run_spiral_protocol_i_clean_baselines.py --help
python scripts/run_uea_paper_protocol.py --help
```

The smoke command uses two iterations and writes only to `outputs/smoke/`.

## Spiral paper protocol

Generate a manifest without starting training:

```bash
python scripts/run_spiral_paper_protocol.py \
  --root . \
  --write-manifest \
  --models transformer latent_ode contiformer hippogp \
  --alphas 0.02 0.2 2.0 \
  --seeds 13 27 42 100 2024 \
  --niters 5000 \
  --noise-std 0.1 \
  --nsample 30 \
  --skip-existing
```

Run one manifest entry by adding `--task-index 0`, or run the full manifest only after reviewing the generated manifest and output root. The clean validation-only runner is separate:

```bash
python scripts/run_spiral_protocol_i_clean_baselines.py --help
```

## UEA paper protocol

The UEA runner uses the official TRAIN/TEST files, observation drops `0.3`, `0.5`, and `0.7`, and seeds `27`, `42`, and `1024`. Start with one dataset and one seed:

```bash
python scripts/run_uea_paper_protocol.py \
  --datasets Heartbeat \
  --drop-ratios 0.3 \
  --seeds 27 \
  --variants baseline_contiformer hippogp_analytic \
  --dry-run
```

To train, remove `--dry-run`. Download the UEA Multivariate 2018 archive into `PhysioPro/data/Multivariate_ts/` before training. The runner never includes raw data or prior result archives.

## Supplemental rebuttal diagnostics

`rebuttal/phase45_public/` contains the sanitized Phase 4.5 supplemental package
from the latest rebuttal release. It includes aggregate CSVs, LaTeX tables,
protocol reports, and analysis scripts for ablation, rank, query/observation
scaling, wall-clock training, time-to-quality, the scoped UEA analysis, and the
corrected HiPPO-only PhysioNet analysis. These are supplemental diagnostics,
not replacements for the primary paper tables. The package explicitly excludes
invalid PhysioNet seed runs, unmatched CRU evidence, blocked UEA datasets,
event raw data, checkpoints, and the protected paper-revision candidate.

## Scope and evidence boundary

This export contains reproducibility code only. It deliberately omits raw data, checkpoints, historical result archives, private remote logs, personal documents, machine-specific paths, and protocol-conflicting evidence. The code does not claim that a local rerun reproduces every submitted table unless the exact data, seed, source commit, hardware, precision, and output manifest are recorded for that run.
