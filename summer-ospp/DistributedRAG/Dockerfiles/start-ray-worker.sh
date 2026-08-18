#!/usr/bin/env bash
set -euo pipefail

args=(ray start --address="${RAY_HEAD_ADDRESS:-ray-head:6379}" --num-cpus="${RAY_WORKER_CPUS:-4}" --block)

if [[ "${RAY_WORKER_GPUS:-0}" != "0" ]]; then
  args+=(--num-gpus="${RAY_WORKER_GPUS}")
fi

if [[ -n "${RAY_CUSTOM_RESOURCES:-}" ]]; then
  args+=(--resources="${RAY_CUSTOM_RESOURCES}")
fi

exec "${args[@]}"
