#!/usr/bin/env bash
# Run inference for all models in DDCF_data/model_order.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="${SCRIPT_DIR}/DDCF_data/model_order.csv"

if [[ ! -f "$CSV" ]]; then
  echo "Error: $CSV not found" >&2
  exit 1
fi

# Skip header, read model_name (second column)
while IFS=',' read -r _ model_name; do
  [[ -z "$model_name" || "$model_name" == "model_name" ]] && continue
  echo "Running inference for: $model_name"
  python infer.py --model_name "$model_name" || {
    echo "Inference failed for $model_name" >&2
    exit 1
  }
done < <(tail -n +2 "$CSV")

echo "Done. Inference completed for all models."
