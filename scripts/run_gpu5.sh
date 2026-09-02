#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_gpu5.sh <train|inference> <experiment>" >&2
  exit 2
fi

mode="$1"
experiment="$2"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1

case "$mode" in
  train)
    exec python3 -u train.py --experiment "$experiment"
    ;;
  inference)
    exec python3 -u inference.py --experiment "$experiment"
    ;;
  *)
    echo "Unknown mode '$mode'. Use train or inference." >&2
    exit 2
    ;;
esac
