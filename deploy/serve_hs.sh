#!/usr/bin/env bash
# Run on hs after download_model.py has verified both checkpoints.
# The current text PP4 configuration still needs live startup/workflow validation.
set -euo pipefail

root=${VOCAGENT_DEPLOY_ROOT:-/data/group6/vocagent}
image=registry.mthreads.com/presale/devtech/vllm_musa@sha256:bec946b86d389888b8af8660537b786833fe41eb7121a3070561ddc648bc6237

case "${1:-}" in
  text)
    model=Qwen3-Next-80B-A3B-Instruct
    name=vocagent-qwen3
    served=qwen3
    port=18080
    memory=0.82
    dtype=bfloat16
    context=16384
    tensor_parallel=1
    pipeline_parallel=4
    ;;
  guard)
    model=Qwen3Guard-Gen-8B
    name=vocagent-qwen3-guard
    served=qwen3-guard
    port=18081
    memory=0.105
    dtype=bfloat16
    context=8192
    tensor_parallel=4
    pipeline_parallel=1
    ;;
  *) echo "Usage: $0 text|guard" >&2; exit 2 ;;
esac

test -f "$root/models/$model/modelscope-download-manifest.json" || {
  echo "Download and verify Qwen/$model before serving it." >&2
  exit 1
}

# Physical GPUs 4-7 belong to group6; GPUs 0-3 run another group's workload.
# Both services shard across the same four GPUs within fixed memory budgets.
exec docker run -d --name "$name" --restart unless-stopped \
  --runtime mthreads --shm-size 16g \
  -e MTHREADS_VISIBLE_DEVICES=4,5,6,7 \
  -e MTHREADS_DRIVER_CAPABILITIES=all \
  -e MUSA_VISIBLE_DEVICES=0,1,2,3 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -p "127.0.0.1:$port:8000" \
  -v "$root/models/$model:/model:ro" \
  --entrypoint python "$image" -m vllm.entrypoints.openai.api_server \
  --model /model --served-model-name "$served" --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size "$tensor_parallel" --pipeline-parallel-size "$pipeline_parallel" --dtype "$dtype" \
  --max-model-len "$context" --max-num-seqs 4 --max-num-batched-tokens 512 \
  --gpu-memory-utilization "$memory" --enforce-eager \
  --block-size 32 --no-enable-prefix-caching --no-async-scheduling \
  --cpu-offload-gb 0 --kv-cache-dtype auto \
  --generation-config auto
