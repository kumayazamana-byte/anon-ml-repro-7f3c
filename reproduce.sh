#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/PhysioPro:${PYTHONPATH:-}"

case "${1:-help}" in
  smoke)
    mkdir -p "$ROOT/outputs/smoke"
    python "$ROOT/SeqML/ContiFormer/spiral.py" \
      --adjoint=1 --visualize=0 --niters=2 --model_name Neural_ODE \
      --noise_a=0.02 --cc=2 --train_dir "$ROOT/outputs/smoke/neuralode" \
      --log_step=1 --nspiral=12 --ntotal=30 --nsample=10 --ntrain=8 --ntest=4 --batch_size=4
    python "$ROOT/SeqML/ContiFormer/spiral.py" \
      --adjoint=1 --visualize=0 --niters=2 --model_name Contiformer \
      --noise_a=0.02 --cc=2 --train_dir "$ROOT/outputs/smoke/contiformer" \
      --log_step=1 --nspiral=12 --ntotal=30 --nsample=10 --ntrain=8 --ntest=4 --batch_size=4
    ;;
  help|*)
    echo "Usage: bash reproduce.sh smoke"
    echo "See README.md for full manifest-generation and training commands."
    ;;
esac
