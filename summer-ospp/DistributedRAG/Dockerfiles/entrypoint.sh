#!/usr/bin/env bash
set -euo pipefail

if [[ -f /root/miniconda3/etc/profile.d/conda.sh && -n "${CONDA_ENV_NAME:-}" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

exec "$@"
