#!/bin/bash
# Per-node Megatron entry for 128-node production (cuda-aligned musa_pretrain_ws128.sh).
# Invoked by dist_run_megatron.sh with WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJ_DIR"

export HOSTFILE="${HOSTFILE:-${PROJ_DIR}/hostfile}"
export RANK="${RANK:-${NODE_RANK:-0}}"
export NNODES="${WORLD_SIZE:?WORLD_SIZE required}"
export MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR required}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export TOKENIZER_PATH="${TOKENIZER_PATH:-${PROJ_DIR}/tokenizer}"
export DATA_PATH="${DATA_PATH:-/home/jd/wangkang/llm_pretrain/data/tkn_ds_the_pile}"
export SAVE_PATH="${SAVE_PATH:-/home/jd/wangkang/llm_pretrain/outputs}"
export LOG_OUTPUT="${LOG_OUTPUT:-/home/jd/wangkang/llm_pretrain/outputs/logs}"
export FOREGROUND="${FOREGROUND:-1}"

export MUSA_PRETRAIN_ENTRY="${MUSA_PRETRAIN_ENTRY:-musa_pretrain_ws128.sh}"

# JoyVideo dist_train bootstrap（TCPStore rendezvous）
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/jd/blake/mudnn0515/mudnn/lib:/usr/local/openmpi/lib:/usr/local/musa/lib:${LD_LIBRARY_PATH:-}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-900000}"
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"
export MUSA_USERQ="${MUSA_USERQ:-1}"
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_MCCL_USE_DEFAULT_STREAM="${TORCH_MCCL_USE_DEFAULT_STREAM:-1}"
export TORCH_MCCL_AVOID_RECORD_STREAMS="${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}"
export TOKENIZERS_PARALLELISM=false
# JoyVideo dist_train.sh 对齐：bond0 socket 建联（跨机架 10.121.x 可达）；RDMA 仍走 IB
export MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-bond0}"
export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
export MCCL_IB_TC="${MCCL_IB_TC:-122}"
export MCCL_NET_SHARED_BUFFERS="${MCCL_NET_SHARED_BUFFERS:-0}"
export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-20971520}"
export MCCL_IB_TIMEOUT="${MCCL_IB_TIMEOUT:-19}"
export MCCL_IB_RETRY_CNT="${MCCL_IB_RETRY_CNT:-7}"
export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_ALGOS="${MCCL_ALGOS:-1}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"

ulimit -n 524288
ulimit -s unlimited
ulimit -c unlimited

echo "dist_train_megatron_ws128: RANK=${RANK} NNODES=${NNODES} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  SAVE_PATH=${SAVE_PATH}"
echo "  ulimit -n=$(ulimit -n)"
echo "  MUSA_PRETRAIN_ENTRY=${MUSA_PRETRAIN_ENTRY}"

exec bash "${PROJ_DIR}/${MUSA_PRETRAIN_ENTRY}"
