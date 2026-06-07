#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: rename to agg_router_2gpu.sh (uses 2 GPUs) and update all references
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="google/gemma-4-26b-a4b-it"
BLOCK_SIZE=64

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated + KV Routing (1 GPUs)" "$MODEL" "$HTTP_PORT"

# run frontend + KV router
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states --dyn-chat-processor vllm --tool-call-parser gemma4 &

# run workers
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
#
# If multiple workers are launched, they must not share the same system/metrics port.
# Use DYN_SYSTEM_PORT{1,2} so tests/launchers can provide a simple numbered port set.
# TODO: use build_vllm_gpu_mem_args to measure VRAM instead of relying on vLLM defaults
#
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE --attention-backend GEMMA4_FLASH_ATTN \
    --max-model-len 250000 --language-model-only --gpu-memory-utilization 0.95 --quantization fp8 \
    --speculative-config='{"model": "google/gemma-4-26b-a4b-it-assistant", "num_speculative_tokens": 4}' \
    --prefix-warmup-file "https://gist.githubusercontent.com/Tushar-ml/d9a8fa4f076f1585b19ebfa7d5c5ca8d/raw/312628f74aec1711e34357488aad6dc2d7692fe4/warmup.json"  \
    --prefix-warmup-parallel \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
# VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
# CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
#     --model $MODEL \
#     --block-size $BLOCK_SIZE \
#     --enforce-eager \
#     --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
