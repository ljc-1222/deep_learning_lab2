#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dl_lab2"
PYTHON_VERSION="3.13"
REQ_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not installed or not in PATH."
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: requirements.txt not found at $REQ_FILE"
  exit 1
fi

# Create env only if missing
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists. Skipping creation."
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
  echo "Installing dependencies from requirements.txt"
  python -m pip install --upgrade pip
  python -m pip install -r "$REQ_FILE"
fi

echo "Current Python: $(python --version)"