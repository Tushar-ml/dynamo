#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Common configuration
MODEL="google/gemma-4-31b-it"
BLOCK_SIZE=64
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

# KVBM tiering defaults for *single-GPU aggregated* serving (this script's default).
#
# With one worker, CPU write-through offload hurts tail latency: every prefix
# cache hit can require a GPU<-CPU onboard (PCIe), which shows up on long /
# queued requests (e.g. output-jitter runs) while BASE stays similar to agg_router.
#
# Default: keep KV on GPU (priority 100 = offload nothing; blocks stay in G1).
# Minimal CPU pool satisfies KVBM init without pinning 40+ GiB host RAM.
#
# For 2+ workers / disagg where KVBM spans GPUs, override before launch:
#   export DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY=0
#   export DYN_KVBM_CPU_CACHE_GB=64   # must be >= GPU KV cache size
: "${DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY:=100}"
: "${DYN_KVBM_CPU_CACHE_GB:=1}"

print_launch_banner "Launching Aggregated + KVBM + KV Routing (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "KVBM CPU offload: disabled (HOST_OFFLOAD_PREFIX_MIN_PRIORITY=${DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY})" \
    "KVBM CPU cache: ${DYN_KVBM_CPU_CACHE_GB} GiB (G1-only mode; raise for multi-GPU)"

# run frontend + KV router
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# run workers with KVBM enabled
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
# Each worker needs unique ZMQ ports to avoid KVBM coordination conflicts
# TODO: use build_vllm_gpu_mem_args to measure VRAM instead of hardcoded fractions
DYN_KVBM_LEADER_ZMQ_PUB_PORT=56001 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56002 \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY=$DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY \
DYN_KVBM_CPU_CACHE_GB=$DYN_KVBM_CPU_CACHE_GB \
CUDA_VISIBLE_DEVICES=0 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE --attention-backend GEMMA4_FLASH_ATTN \
    --max-model-len 32000 --language-model-only --gpu-memory-utilization 0.95 --quantization fp8 \
    --speculative-config='{"model": "google/gemma-4-31b-it-assistant", "num_speculative_tokens": 4}' \
    --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# DYN_KVBM_LEADER_ZMQ_PUB_PORT=56003 \
# DYN_KVBM_LEADER_ZMQ_ACK_PORT=56004 \
# VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
# DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY=0 \
# DYN_KVBM_CPU_CACHE_GB=64 \
# CUDA_VISIBLE_DEVICES=1 \
#     python3 -m dynamo.vllm \
#     --model $MODEL \
#     --enforce-eager \
#     --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}' \
#     --gpu-memory-utilization 0.4 \
#     --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
