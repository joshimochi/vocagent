#!/usr/bin/env bash
# Execute on hs after setup; serialize the shared four-GPU allocation.
set -euo pipefail
root=${VOCAGENT_DEPLOY_ROOT:-/data/group6/vocagent}
container=${VOCAGENT_CONTAINER:-vocagent-local-a406}
app=${VOCAGENT_APP_DIR:-$root/run-a406/app}
venv=${VOCAGENT_VENV:-$root/run-a406/venv}
mkdir -p "$root/logs"
if (($# == 0)); then
  echo 'Usage: bash deploy/run_hs.sh WORD [WORD ...] --output-dir DIRECTORY' >&2
  exit 2
fi
if [[ $(docker inspect -f '{{.State.Running}}' "$container") != true ]]; then
  docker start "$container" >/dev/null
fi
# Pass a fixed allow-list; never forward the host's unrelated environment/secrets.
environment=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e "PYTHONPATH=$app"
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8
  -e "PATH=$root/run-a406/media/current/bin:/usr/local/musa/bin:/usr/local/bin:/usr/bin:/bin")
for variable in VOCAGENT_DEPLOY_ROOT QWEN_VL_MODEL QWEN_GUARD_MODEL_PATH QWEN_ASR_MODEL \
  QWEN_ALIGNER_MODEL SHIELDGEMMA_MODEL MINIMAX_H3_MODEL MINIMAX_H3_WIDTH \
  MINIMAX_H3_HEIGHT MINIMAX_H3_SECONDS MINIMAX_H3_STEPS MINIMAX_H3_PYTHON \
  MINIMAX_H3_TIMEOUT_SECONDS; do
  if [[ -v $variable ]]; then environment+=(-e "$variable=${!variable}"); fi
done
# The container holds the lock through Python exit, even if SSH disconnects.
# No daemon/API, weight offloading, or persistent second model. Docker does not
# expose inference ports. The Python runtime evicts the previous model family.
exec docker exec "${environment[@]}" -w "$app" "$container" \
  flock -n "$root/vocagent-gpu.lock" "$venv/bin/python" -m vocagent "$@"
