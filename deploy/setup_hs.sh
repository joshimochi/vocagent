#!/usr/bin/env bash
# Run on hs from the uploaded app directory. Never modifies the host GPU stack.
set -euo pipefail
root=${VOCAGENT_DEPLOY_ROOT:-/data/group6/vocagent}
run=$root/run-a406
container=${VOCAGENT_CONTAINER:-vocagent-local-a406}
image=registry.mthreads.com/presale/devtech/vllm_musa@sha256:bec946b86d389888b8af8660537b786833fe41eb7121a3070561ddc648bc6237
app=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$run" "$root/logs" "$root/models"
if ! docker container inspect "$container" >/dev/null 2>&1; then
  docker run -d --name "$container" --runtime mthreads --shm-size 16g \
    -e MTHREADS_VISIBLE_DEVICES=4,5,6,7 -e MTHREADS_DRIVER_CAPABILITIES=all \
    -e MUSA_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$root:$root" -v /data/group6/minimax-h3:/data/group6/minimax-h3 \
    --entrypoint /bin/bash "$image" -lc 'exec sleep infinity'
fi
if [[ $(docker inspect -f '{{.State.Running}}' "$container") != true ]]; then
  docker start "$container"
fi
# venv's ensurepip is absent in the vendor image. Bootstrap virtualenv only into
# our bind-mounted prefix; both environments inherit the installed MUSA torch.
docker exec "$container" python -m pip install --target "$run/bootstrap" virtualenv
docker exec -e "PYTHONPATH=$run/bootstrap" "$container" \
  python -m virtualenv --system-site-packages "$run/venv"
docker exec -e "PYTHONPATH=$run/bootstrap" "$container" \
  python -m virtualenv --system-site-packages "$run/h3-venv"
printf 'torch==2.9.0\n' > "$run/musa-constraints.txt"
docker exec "$container" "$run/venv/bin/python" -m pip install \
  -c "$run/musa-constraints.txt" -r "$app/deploy/requirements-speech.txt"
docker exec "$container" "$run/h3-venv/bin/python" -m pip install \
  -c "$run/musa-constraints.txt" -r "$app/deploy/requirements-h3.txt"
# Transformers video processors require torchvision. The CPU wheel handles
# preprocessing only; --no-deps preserves vendor MUSA PyTorch and GPU inference.
for environment in "$run/venv" "$run/h3-venv"; do
  docker exec "$container" "$environment/bin/python" -m pip install --no-deps \
    --index-url https://download.pytorch.org/whl/cpu torchvision==0.24.0+cpu
done
# Extract media libraries into their own prefix: no apt install/driver overwrite.
docker exec "$container" bash "$app/deploy/install_media.sh" "$run/media"
for model in Qwen3-VL-32B-Instruct Qwen3Guard-Gen-8B Qwen3-ASR-1.7B Qwen3-ForcedAligner-0.6B; do
  python3 "$app/deploy/download_model.py" "Qwen/$model" "$root/models/$model" --workers 8
done
for model in "$root/models/shieldgemma-2-4b-it/config.json" \
  /data/group6/minimax-h3/models/MiniMax-H3/modular_model_index.json; do
  test -f "$model" || { echo "Required existing checkpoint is missing: $model" >&2; exit 1; }
done
printf 'Setup finished. Run bash %s/deploy/run_hs.sh WORDS --output-dir DIRECTORY\n' "$app"
