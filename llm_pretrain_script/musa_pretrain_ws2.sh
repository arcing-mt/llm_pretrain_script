#!/bin/bash
# 双机 MUSA 预训练脚本（ws2 验证专用，2026-07-04 已跑通）
#
# 设计原则：在 16 卡（2×8）约束下，除「模型结构 / 并行拓扑 / 集群规模 / 路径 / MUSA 适配」
# 外，其余配置尽可能与 cuda_pretrain.sh 保持一致。
# 128 机生产配置见 musa_pretrain_ws128.sh（勿在本脚本改并行/模型去凑 128 机）。
#
# 用法（经 dist_train_megatron.sh / auto_fault_manager）:
#   LOG_NAME=ws2_verify TRAINING_STEPS=5 bash musa_pretrain_ws2.sh
#
# 仅通信探测（非 MoE，勿用于生产对齐验证）:
#   PROFILE=debug_comm TRAINING_STEPS=2 bash musa_pretrain_ws2.sh
#
# 兼容旧名: musa_pretrain_multi2.sh → 转发本脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"
PROFILE=${PROFILE:-moe}   # 默认 moe=对齐 cuda MoE；debug_comm=仅测双机通信

# ---------------------------------------------------------------------------
# hostfile → 集群拓扑
# 原 cuda_pretrain.sh: NNODES=256, MASTER_ADDR=service-train-v3-s1-0
# 现: 从 hostfile 读取，双机验证 NNODES=2
# ---------------------------------------------------------------------------
if [ ! -f "${HOSTFILE}" ]; then
    echo "ERROR: hostfile 不存在: ${HOSTFILE}" >&2
    exit 1
fi

mapfile -t HOSTS < <(grep -v '^[[:space:]]*#' "${HOSTFILE}" | awk '{print $1}' | grep -v '^$')
NNODES=${NNODES:-${#HOSTS[@]}}
if [ "${NNODES}" -lt 1 ]; then
    echo "ERROR: hostfile 无有效节点: ${HOSTFILE}" >&2
    exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-${HOSTS[0]}}
# 原 cuda_pretrain.sh: MASTER_PORT=22
# 现: 29500 — torchrun 分布式 rendezvous 端口，22 为 SSH 端口无法用于 NCCL/MCCL 建联
MASTER_PORT=${MASTER_PORT:-29500}

LOCAL_IP=$(ip -4 -o addr show bond0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
NODE_RANK=${RANK:-${NODE_RANK:-}}
if [ -z "${NODE_RANK}" ]; then
    rank=0
    for ip in "${HOSTS[@]}"; do
        if [ "${ip}" = "${LOCAL_IP}" ]; then
            NODE_RANK=${rank}
            break
        fi
        rank=$((rank + 1))
    done
fi
if [ -z "${NODE_RANK}" ] || [ "${NODE_RANK}" -ge "${NNODES}" ]; then
    echo "ERROR: 无法确定 NODE_RANK (LOCAL_IP=${LOCAL_IP}, hostfile=${HOSTFILE})" >&2
    exit 1
fi

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}   # 原: 8（不变）
export GPU_NUM=$((${GPUS_PER_NODE} * ${NNODES}))
export WORLD_SIZE=$((${GPUS_PER_NODE} * ${NNODES}))
export NODE_RANK MASTER_ADDR MASTER_PORT NNODES

# ---------------------------------------------------------------------------
# 环境变量（cuda_pretrain.sh → MUSA/MCCL 映射）
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export LD_LIBRARY_PATH=/usr/local/musa/lib:${LD_LIBRARY_PATH:-}
export MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MUSA_KERNEL_TIMEOUT=${MUSA_KERNEL_TIMEOUT:-3200000}
export MUSA_BLOCK_SCHEDULE_MODE=${MUSA_BLOCK_SCHEDULE_MODE:-1}
export ACCELERATOR_BACKEND="musa"

# 原 cuda: NCCL_SOCKET_IFNAME=eth0
# 现: 8×400G RoCE 网卡（172.21.x）；hostfile/SSH 仍走 bond0(10.121.32.x)
export MCCL_SOCKET_IFNAME=${MCCL_SOCKET_IFNAME:-ens12np0,ens11np0,ens13np0,ens14np0,ens16np0,ens15np0,ens17np0,ens18np0}
export MCCL_PXN_DISABLE=${MCCL_PXN_DISABLE:-0}                    # 原: 0
export MCCL_CROSS_NIC=${MCCL_CROSS_NIC:-1}                        # 原: 1
export MCCL_IB_QPS_PER_CONNECTION=${MCCL_IB_QPS_PER_CONNECTION:-8}  # 原: 8
export MCCL_TIMEOUT=${MCCL_TIMEOUT:-3600}                         # 原: 3600
export MCCL_IB_TIMEOUT=${MCCL_IB_TIMEOUT:-22}                     # 原: 22
export MCCL_SOCKET_NTHREADS=${MCCL_SOCKET_NTHREADS:-8}            # 原: 8
export MCCL_IB_DISABLE=${MCCL_IB_DISABLE:-0}                      # 原 cuda: 0（启用 IB/RDMA）
export MCCL_DEBUG=${MCCL_DEBUG:-WARN}                           # 原 cuda: INFO；INFO 会刷屏 mcclEnqueueCheck/Channel 等
export MCCL_PROTOS=${MCCL_PROTOS:-2}
export MCCL_CHECK_POINTERS=${MCCL_CHECK_POINTERS:-0}
export MCCL_ALGOS=${MCCL_ALGOS:-1}
export MCCL_BUFFSIZE=${MCCL_BUFFSIZE:-20480000}
export MCCL_LIB=${MCCL_LIB:-/usr/local/musa/lib/libmccl.so}
# 原 cuda: NCCL_IB_HCA=mlx5_gdr_0,...,mlx5_gdr_7
# 现: 8×400G mlx5（ibdev2netdev 对应 ens*）；gdr 别名不存在
export MCCL_IB_HCA=${MCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11}
export MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX:-3}                  # 原 cuda: 1；本机 RoCEv2 IP 在 index 3

# 原 cuda: CUDA_DEVICE_MAX_CONNECTIONS=32
# 现: 1 — Megatron 在使用 TP/CP 时要求此值为 1（MUSA 环境实测，设 32 会 AssertionError）
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

export PYTORCH_MUSA_ALLOC_CONF=${PYTORCH_MUSA_ALLOC_CONF:-"expandable_segments:True"}  # 原: PYTORCH_CUDA_ALLOC_CONF
export TORCH_MCCL_AVOID_RECORD_STREAMS=${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}
export TORCH_MCCL_TRACE_BUFFER_SIZE=${TORCH_MCCL_TRACE_BUFFER_SIZE:-1000000}  # 原: TORCH_NCCL_TRACE_BUFFER_SIZE

export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-8}   # 原: 8
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-8}   # 原: 8
export NVTE_DP_AMAX_REDUCE_INTERVAL=${NVTE_DP_AMAX_REDUCE_INTERVAL:-0}   # 原: 0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}  # 原: 1
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-0}                           # 原: 0
export NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-1}                           # 原: 1
export NVTE_EXT_MARGIN_SM=${NVTE_EXT_MARGIN_SM:-20}                    # 原: 20
export NVTE_DEBUG=${NVTE_DEBUG:-0}                                     # 原: 1；验证时降为 0 减少日志
export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}                         # 原: 2
# 原 cuda: NVTE_NORM_*_USE_CUDNN / CUDNN_LOG* — MUSA 无 cuDNN，省略

export USE_MUSA_MOE=${USE_MUSA_MOE:-1}                                 # 原 cuda 无，musa_pretrain 新增
export USE_DEEPEP_ACE=${USE_DEEPEP_ACE:-0}                             # 原 musa_pretrain: 1；alltoall 模式下关闭
export USE_RECOMPUTE_VARIANCE=${USE_RECOMPUTE_VARIANCE:-0}
export ENABLE_D2H_IN_PERMUTATION=${ENABLE_D2H_IN_PERMUTATION:-0}
export NO_LOSS_REDUCE=${NO_LOSS_REDUCE:-1}                             # 原 cuda: 0；MUSA patch loss 上报格式不兼容标量写入

# ---------------------------------------------------------------------------
# 代码路径
# 原 cuda: MCORE_PATH=/mnt/workspace/jdcloud/Megatron-LM, 直接 pretrain_gpt.py
# 现: 容器内 Megatron pretrain_gpt.py + launcher 注入 musa_patch（overlap-comm 需 core 版 forward_step）
# ---------------------------------------------------------------------------
MCORE_PATH=${MCORE_PATH:-/home/Megatron-LM}
PATCH_HOME=${PATCH_HOME:-/home/megatron-lm-musa-patch}
LAUNCHER=${SCRIPT_DIR}/pretrain_gpt_musa_launcher.py
export MCORE_PATH
export PRETRAIN_SCRIPT=${PRETRAIN_SCRIPT:-${PATCH_HOME}/examples/llama3/pretrain_gpt.py}
# 原 cuda: ${MCORE_PATH}/pretrain_gpt.py
# 现: patch 版 — 兼容 musa_patch loss 上报；core 版 MoE 训练日志 Tensor 无法转 Scalar
export PYTHONPATH=${MCORE_PATH}:${PATCH_HOME}:${PYTHONPATH:-}

if [ ! -d "${MCORE_PATH}/build" ]; then
    pushd "${MCORE_PATH}" >/dev/null
    python setup.py build_ext --inplace
    popd >/dev/null
fi

# ---------------------------------------------------------------------------
# 并行策略
# 原 cuda: TP=2, PP=8, CP=1, EP=64, WORLD_SIZE=2048 → DP=2
# 现: TP=1, PP=1, CP=1, EP=4, WORLD_SIZE=16 → DP=4
#     TP 2→1：MUSA patch 在 TP+EP+sequence_parallel 下 backward 梯度数量错误；
#             Megatron 要求 TP+EP 时启用 sequence_parallel，与 patch 冲突，故降为 TP=1
#     PP 8→1、EP 64→4：16 卡等比缩减
# ---------------------------------------------------------------------------
TP=1
PP=1
CP=1
EP=4
MTP_LAYERS=0
MTP_LOSS=0.1

# 原 cuda: LAYOUT="Et|(tt|)*30L"（61 层 + PP=8）
# 现: 不传 --pipeline-model-parallel-layout — Et|(tt|)*7L 解析为 9 stages，无法整除 PP=2；
#     改由 Megatron 对 16 层 / PP=2 做默认均分（各 rank 8 层）
LAYOUT=""

# ---------------------------------------------------------------------------
# 训练超参（与 cuda_pretrain.sh 对齐，可通过环境变量覆盖）
# ---------------------------------------------------------------------------
MICRO_BATCH=1                                                        # 原: 1
GLOBAL_BATCH=${GLOBAL_BATCH:-$((NNODES * 128))}           # 原: NNODES*128
SEQ_LENGTH=${SEQ_LENGTH:-4096}                                         # 原 cuda: 4096；验证可设 2048/512
DECAY_STEPS=100000                                                     # 原: 100000
TRAINING_STEPS=${TRAINING_STEPS:-100000}                             # 原: 100000；验证时可设 5
SAVE_INTERVAL=100000                                                   # 原: 100000
LR_WARMUP_INIT=0.0
WARMUP_STEPS=1000                                                      # 原: 1000
LR=2e-4
LR_MIN=2e-5
DECAY_STYLE=cosine
ADAM_BETA1=0.9
ADAM_BETA2=0.95
LB_RATE=1e-4
RB_RATE=1e-3
INIT_STD=0.006

# ---------------------------------------------------------------------------
# 数据 / 输出路径
# 原 cuda: TOKENIZER/DATA/SAVE/LOG 均在 /mnt/workspace/...
# 现: NFS 本地验证路径；DATA_PATH 目录结构不同见 DATA PROCESS 注释
# ---------------------------------------------------------------------------
BASE=/mnt/code/llm_pretrain
VERSION=test-multi2
STAGE=${LOG_NAME:-"moe-16gpu-$(date +%Y%m%d_%H%M%S)"}
TOKENIZER_PATH=${TOKENIZER_PATH:-${BASE}/tokenizer}
SAVE_PATH=${SAVE_PATH:-/home/jd/wangkang/llm_pretrain/outputs}
LOG_OUTPUT=${LOG_OUTPUT:-/home/jd/wangkang/llm_pretrain/outputs/logs}
DATA_PATH=${DATA_PATH:-/home/jd/wangkang/llm_pretrain/data/tkn_ds_the_pile}

# ---------------------------------------------------------------------------
# 模型结构（MoE + MLA，对齐 cuda_pretrain.sh，仅缩小到 16 卡可运行规模）
# ---------------------------------------------------------------------------
if [ "${PROFILE}" = "debug_comm" ]; then
    # 非生产配置：仅用于双机通信探测，不对齐 cuda 模型
    echo "WARNING: PROFILE=debug_comm 仅用于通信探测，非 cuda 对齐配置" >&2
    PP=1
    EP=1
    LAYOUT=""
    SEQ_LENGTH=512
    TRAINING_STEPS=${TRAINING_STEPS:-2}
    GLOBAL_BATCH=${GLOBAL_BATCH:-$((NNODES * 8))}
    WARMUP_STEPS=2
    DECAY_STEPS=100
    SAVE_INTERVAL=1000
    export PRETRAIN_SCRIPT=${PATCH_HOME}/examples/llama3/pretrain_gpt.py
    export USE_MUSA_MOE=0
    export USE_DEEPEP_ACE=0
    export NO_LOSS_REDUCE=1
    unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN
    ADD_NETWORK_SIZE_ARGS=(
        --num-layers 4
        --hidden-size 1024
        --ffn-hidden-size 2752
        --num-attention-heads 16
        --kv-channels 64
        --position-embedding-type rope
        --rotary-base 10000
        --rotary-percent 1.0
        --make-vocab-size-divisible-by 128
        --normalization RMSNorm
        --norm-epsilon 1e-6
        --swiglu
        --untie-embeddings-and-output-weights
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --clip-grad 1.0
        --weight-decay 0.1
        --init-method-std ${INIT_STD}
        --disable-bias-linear
        --attention-backend unfused
        --transformer-impl transformer_engine
        --no-persist-layer-norm
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )
else
    ADD_NETWORK_SIZE_ARGS=(
        --num-layers 16                                           # 原: 61 — 16 卡显存无法承载 61 层生产模型
        --hidden-size 2048                                        # 原: 7168
        --ffn-hidden-size 5632                                    # 原: 18432
        --num-attention-heads 16                                  # 原: 128
        --kv-channels 128                                         # 原: 128
        --position-embedding-type rope
        --rotary-base 10000
        --rotary-percent 1.0
        --rope-type rope                                          # 原 cuda: 未显式设置（MLA 默认 yarn）；配合 --no-rope-fusion 需显式指定 rope
        --make-vocab-size-divisible-by 128                        # 原: 3232 — 本地 tokenizer 词表较小
        --normalization RMSNorm
        --norm-epsilon 1e-6
        --swiglu
        --untie-embeddings-and-output-weights
        --multi-latent-attention
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --clip-grad 1.0
        --weight-decay 0.1
        --qk-layernorm
        --num-experts 16                                          # 原: 256 — 需整除 EP=4，且 16 卡可承载
        --manual-gc
        --manual-gc-interval 5
        --moe-layer-freq "([0]*1+[1]*15)"                        # 原: "([0]*3+[1]*58)" — 等比：1 dense + 15 moe
        --moe-ffn-hidden-size 512                                 # 原: 2048
        --moe-shared-expert-intermediate-size 512                 # 原: 2048
        --moe-router-load-balancing-type seq_aux_loss
        --moe-router-topk 2                                       # 原: 8 — 缩小激活专家数以适配验证集群
        --moe-router-pre-softmax
        --moe-grouped-gemm
        --moe-router-group-topk 2                                 # 原: 4
        --moe-router-num-groups 2                                 # 原: 8
        --moe-router-topk-scaling-factor 2.5
        --moe-router-score-function sigmoid
        --moe-router-enable-expert-bias
        --q-lora-rank 512                                         # 原: 1536
        --kv-lora-rank 128                                        # 原: 512
        --qk-head-dim 128
        --qk-pos-emb-head-dim 64
        --v-head-dim 128
        --init-method-std ${INIT_STD}
        --attention-backend flash
        --disable-bias-linear
        --moe-router-dtype fp32
        --transformer-impl transformer_engine
        --use-flash-attn
        --no-rope-fusion                                        # 原 cuda: 未显式设置（默认 apply_rope_fusion=True）
                                                                    # MLA+标准 RoPE 在 MUSA Megatron 会断言失败，需显式关闭
        # 原 cuda: --pipeline-model-parallel-layout Et|(tt|)*30L
        # 未传 layout — 见上方 LAYOUT 注释（9 stages 无法整除 PP=2）
        --cross-entropy-loss-fusion
        --cross-entropy-fusion-impl native
        --moe-permute-fusion
        # 原 cuda: --overlap-moe-expert-parallel-comm
        # 未启用 — MUSA patch 下 LinearWithGradAccumulation 反向梯度数量不匹配（expected 9 got 8）
        # 原 cuda: --delay-wgrad-compute
        # 未启用 — 当前容器 TE <2.3.0 不支持；且依赖 overlap-moe-expert-parallel-comm
        --moe-token-dispatcher-type alltoall                      # 原 cuda: flex — MUSA 验证环境 flex+deepep 梯度报错，暂改 alltoall
        # 原 cuda: --moe-enable-deepep
        # 未启用 — 与 alltoall dispatcher 配套；flex+deepep 在 MUSA 验证中触发梯度数量错误
        # 原 cuda: --sequence-parallel
        # 未启用 — musa_patch LinearWithGradAccumulation backward 在 sequence_parallel 下返回 8 个梯度而非 9
        --moe-router-force-load-balancing
        # 原 cuda: --moe-router-fusion
        # 未启用 — musa_patch 导入后 Megatron 参数表不包含此选项（裸 pretrain_gpt.py 有，经 launcher 不可用）
        # 原 cuda: --moe-shared-expert-compute-before-router
        # 未启用 — 当前 /home/Megatron-LM 版本不支持该 jdcloud 定制参数
    )
fi

if [ "$MTP_LAYERS" -gt 0 ]; then
    ADD_NETWORK_SIZE_ARGS=(
        ${ADD_NETWORK_SIZE_ARGS[@]}
        --mtp-num-layers ${MTP_LAYERS}
        --mtp-loss-scaling-factor ${MTP_LOSS}
    )
fi

# ---------------------------------------------------------------------------
# DATA PROCESS（对齐 cuda_pretrain.sh 加权 blend 逻辑）
# 原 cuda: DATA_PATH=/mnt/workspace/data/merged，文件名如 *-merge.bin
# 现: 本地 DATA_PATH 为 tkn_ds_the_pile/*_text_document.bin，文件名不匹配 STAGE1_DATA
#     → 先走 cuda 同名匹配；若无匹配则 fallback 等权使用全部 *.bin
# ---------------------------------------------------------------------------
declare -A STAGE1_DATA=(
    ["2-3-merge.bin"]=86.683163068
    ["3-4-merge.bin"]=59.885228928
    ["CC-MAIN-2021-04-merge.bin"]=16.826519427
    ["CC-MAIN-2021-10-merge.bin"]=13.837581715
    ["CC-MAIN-2021-17-merge.bin"]=14.980002313
    ["CC-MAIN-2021-21-merge.bin"]=9.779054111
    ["CC-MAIN-2021-25-merge.bin"]=12.707104188
    ["CC-MAIN-2021-31-merge.bin"]=18.312108025
    ["CC-MAIN-2021-39-merge.bin"]=16.28938315
    ["CC-MAIN-2021-43-merge.bin"]=18.801128405
    ["CC-MAIN-2021-49-merge.bin"]=12.684432821
    ["CC-MAIN-2022-05-merge.bin"]=16.324092216
    ["CC-MAIN-2022-21-merge.bin"]=20.576187199
    ["CC-MAIN-2022-27-merge.bin"]=13.207847354
    ["CC-MAIN-2022-33-merge.bin"]=9.65958263
    ["CC-MAIN-2023-06-merge.bin"]=19.724194067
    ["CC-MAIN-2023-14-merge.bin"]=16.33230536
    ["CC-MAIN-2023-23-merge.bin"]=21.760590368
    ["CC-MAIN-2024-22-merge.bin"]=14.299276047
    ["CC-MAIN-2024-26-merge.bin"]=13.019718751
    ["CC-MAIN-2024-30-merge.bin"]=12.693607986
    ["CC-MAIN-2024-33-merge.bin"]=10.86458841
    ["CC-MAIN-2024-38-merge.bin"]=13.383161961
    ["CC-MAIN-2024-42-merge.bin"]=11.09886149
    ["CC-MAIN-2024-46-merge.bin"]=12.230566164
    ["CC-MAIN-2024-51-merge.bin"]=12.740120855
)
FILE_NAMES=()
while IFS= read -r file; do
    FILE_NAMES+=("${file}")
done < <(find "$DATA_PATH" -type f -name "*.bin" 2>/dev/null | sort)
DATA_PATHS=()
for file_name in "${FILE_NAMES[@]}"; do
    file_path=$file_name
    file_name_only="${file_path#"$DATA_PATH/"}"
    if [[ -v STAGE1_DATA["$file_name_only"] ]]; then
        weight=${STAGE1_DATA["$file_name_only"]}
    else
        continue
    fi
    length=${#file_path}
    DATA_PATHS+=("${weight} ${file_path:0:$((length - 4))}")
done
if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    # fallback: 本地验证数据无 merge.bin 命名，等权使用可用分片
    # 原 cuda STAGE1_DATA 中所有 merge.bin 均有配套 idx；本地 02_text_document 缺 idx，需过滤
    echo "NOTE: STAGE1_DATA 无匹配文件，fallback 等权加载 ${DATA_PATH} 下 bin+idx 成对分片" >&2
    for file_name in "${FILE_NAMES[@]}"; do
        file_path=$file_name
        length=${#file_path}
        prefix="${file_path:0:$((length - 4))}"
        idx_path="${prefix}.idx"
        if [ -f "${idx_path}" ]; then
            DATA_PATHS+=("1.0 ${prefix}")
        fi
    done
fi
if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    echo "ERROR: 未找到可用数据，请检查 DATA_PATH=${DATA_PATH}" >&2
    exit 1
fi
DATA_PATH_ARGUMENT=$(printf "%s " "${DATA_PATHS[@]}")
echo "DATA_PATH_ARGUMENT=${DATA_PATH_ARGUMENT}"

if [ ! -d "$SAVE_PATH" ]; then
  mkdir -p "$SAVE_PATH"
fi

EXP_NAME=$VERSION/$STAGE
SAVE_PATH=$SAVE_PATH/$EXP_NAME
LOG_OUTPUT=$LOG_OUTPUT/$EXP_NAME
mkdir -p $LOG_OUTPUT

# ---------------------------------------------------------------------------
# 分布式 / 训练 / 数据 / 日志参数（对齐 cuda_pretrain.sh）
# ---------------------------------------------------------------------------
DISTRIBUTED_ARGS=(
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    --expert-tensor-parallel-size 1
    --expert-model-parallel-size ${EP}
    --use-distributed-optimizer
)

TRAINING_ARGS=(
    --use-mcore-models
    --enable-experimental                                      # 原 cuda 未显式设置；moe-enable-deepep/flex dispatcher 依赖此 flag
    --micro-batch-size ${MICRO_BATCH}
    --global-batch-size ${GLOBAL_BATCH}
    --train-iters ${TRAINING_STEPS}
    --no-check-for-nan-in-loss-and-grad
    --max-position-embeddings ${SEQ_LENGTH}
    --lr-decay-iters ${DECAY_STEPS}
    --lr-warmup-iters ${WARMUP_STEPS}
    --lr-warmup-init ${LR_WARMUP_INIT}
    --lr ${LR}
    --min-lr ${LR_MIN}
    --lr-decay-style ${DECAY_STYLE}
    --adam-beta1 ${ADAM_BETA1}
    --adam-beta2 ${ADAM_BETA2}
    --moe-aux-loss-coeff ${LB_RATE}
    --moe-router-bias-update-rate ${RB_RATE}
    --no-gradient-accumulation-fusion                          # 原 cuda: 未设置；MUSA patch 反向梯度数量不匹配（expected 9 got 8）
    --eval-iters 0
    --eval-interval ${SAVE_INTERVAL}
    --save ${SAVE_PATH}
    --save-interval ${SAVE_INTERVAL}
    --init-method-std ${INIT_STD}
)

if [ "${PROFILE}" = "debug_comm" ]; then
    TRAINING_ARGS+=(
        --no-gradient-accumulation-fusion   # debug_comm 专用：规避 MUSA patch 梯度数量不匹配
    )
fi

DATA_ARGS=(
    --seq-length ${SEQ_LENGTH}
    --data-cache-path ${SAVE_PATH}/cache
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_PATH}
    --data-path ${DATA_PATH_ARGUMENT}
    --split 100,0,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 6                          # 原 cuda: 6
)

LOGGING_ARGS=(
    # 原 cuda: 以下 tensorboard 相关参数全开
    # 现: 关闭 tensorboard — musa_patch training_log 写入标量时维度报错 (size:2 vs 0-dim)
    --log-throughput
    --log-interval 1
    --logging-level 40
    --bf16
)
if [ "${ENABLE_TENSORBOARD:-0}" = "1" ]; then
    LOGGING_ARGS+=(
        --log-timers-to-tensorboard
        --log-memory-to-tensorboard
        --log-num-zeros-in-grad
        --log-params-norm
        --log-validation-ppl-to-tensorboard
        --tensorboard-dir ${SAVE_PATH}/tensorboard
        --tensorboard-log-interval 1
        --moe-per-layer-logging
    )
fi

FILE=${SAVE_PATH}/latest_checkpointed_iteration.txt
if [ -f "$FILE" ]; then
    INPUT=(--load ${SAVE_PATH})
else
    INPUT=()
fi

echo "========================================"
echo "MUSA ws2 验证训练 (rank ${NODE_RANK}/${NNODES})"
echo "  PROFILE    : ${PROFILE}"
echo "  LOCAL_IP   : ${LOCAL_IP}"
echo "  MASTER     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  WORLD_SIZE : ${WORLD_SIZE}"
echo "  TP/PP/EP   : ${TP}/${PP}/${EP}"
echo "  SEQ_LENGTH : ${SEQ_LENGTH}"
echo "  GLOBAL_BS  : ${GLOBAL_BATCH}"
echo "  TRAIN_ITERS: ${TRAINING_STEPS}"
echo "  LOG        : ${LOG_OUTPUT}/output_rank${NODE_RANK}.log"
echo "========================================"

# 原 cuda: nohup torchrun ... pretrain_gpt.py
# 现: 经 LAUNCHER 注入 MUSA patch；FOREGROUND=1 可前台调试
if [ "${FOREGROUND:-0}" = "1" ]; then
    torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ${LAUNCHER} \
        ${DISTRIBUTED_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${ADD_NETWORK_SIZE_ARGS[@]} \
        ${LOGGING_ARGS[@]} \
        ${INPUT[@]} 2>&1 | tee $LOG_OUTPUT/output_rank${NODE_RANK}.log
else
    nohup torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ${LAUNCHER} \
        ${DISTRIBUTED_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${ADD_NETWORK_SIZE_ARGS[@]} \
        ${LOGGING_ARGS[@]} \
        ${INPUT[@]} > $LOG_OUTPUT/output_rank${NODE_RANK}.log 2>&1 &
    echo "已后台启动 torchrun, PID=$!, 日志: $LOG_OUTPUT/output_rank${NODE_RANK}.log"
fi
