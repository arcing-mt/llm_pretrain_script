# Standalone TE TN GM6 wgrad

本文记录当前仓库中的 GM6 接入、启动方式和已知限制。GM6 是 MoE expert
weight-gradient 的可选路径；它不替换 MATE 的 fprop/dgrad，也不改变普通
Transformer Engine grouped GEMM 的默认路径。

## 代码路径

- `musa_patch/tn_gm6/tn_gm6.cpp`：C++/MUSA 扩展，调用 muDNN routed
  `AsmKernelTCEGroupGemm`，计算 `grad_output.T @ input`，输入 BF16、输出 FP32。
- `musa_patch/tn_gm6/_tn_gm6.so`：当前集群 ABI 下已构建的扩展。训练运行时不需要
  muDNN 源码树、sparse-MoE checkout 或生成 kernel library，但仍需要节点上的
  PyTorch-MUSA、Transformer Engine 和对应 MUSA 运行时库。
- `musa_patch/tn_gm6/loader.py`：延迟加载及 Python 调用封装。
- `musa_patch/te_tn_gm6.py`：包装 TE 的三个 `general_grouped_gemm` Python 入口，
  命中条件时转到 GM6，否则原样调用 TE。
- `musa_patch/mate_grouped_gemm.py`：MATE 路径下可选地直接调用同一个 GM6 wgrad。
- `musa_patch/__init__.py`：通过环境变量预加载扩展并安装 wrapper。预加载很重要，
  因为 GM6 和 TE 可能注册重叠的 ASM dispatcher symbol。

## 必须遵守的导入顺序

`musa_patch.tn_gm6._tn_gm6` 必须先于任何 `transformer_engine` 模块加载。这不是普通的
Python import 风格问题：GM6 与 TE 包含重叠的 ASM dispatcher symbol。如果 TE 先加载，
随后在第一次 wgrad 中懒加载 GM6，GM6 kernel 可能解析到 TE 已驻留的 dispatcher，造成
“单 kernel 测试能跑、完整模型路径报错或行为异常”的差异。

因此 `musa_patch/__init__.py` 会在 `patch_before_import_megatron()` 之前检查
`TE_TN_GM6_WGRAD` 和 `MATE_TN_GM6_WGRAD`；任一个为 1 都立即 preload `_tn_gm6.so`。
不要把这个加载移动到 `te_tn_gm6.py` wrapper、MATE backward 或第一次 GEMM 调用里。
训练入口也必须先 `import musa_patch`，再导入 Megatron/Transformer Engine；仓库现有
Megatron 启动方式满足该顺序。

## 两种启用方式

### TE grouped wgrad（推荐作为独立验证）

```bash
export MATE_GROUPED_GEMM=0
export TE_TN_GM6_WGRAD=1
```

这条路径直接拦截 TE `general_grouped_gemm(layout="NT", grad=True)`，不依赖 MATE。
未命中 GM6 条件的调用会回退到 TE，因此可以先用短步数 smoke test 验证加载和数值。

### MATE fprop/dgrad + GM6 wgrad

```bash
export MATE_GROUPED_GEMM=1
export MATE_USE_MAIN_GRAD=1
export MATE_TN_GM6_WGRAD=1
```

此时 MATE 负责 fprop/dgrad，GM6 直接把 wgrad 写入每个 expert 的 FP32
`weight.main_grad`。要求所有节点上的 `mate`/`mate-mubin` 版本匹配；GM6 本身仍是
独立扩展。若某个 batch 有空 expert，MATE 直连 GM6 会放弃该调用并回退 TE；TE
wrapper 会按非空 expert 的连续 runs 尝试 GM6。

## 命中条件

GM6 只接受：

- `layout="NT"`、`grad=True`，且不是 GELU/bias/single-output/D_dtype 特殊路径；
- A/B 为 MUSA BF16、输出为连续 FP32；
- 每个 active expert 的输入和 grad-output 是同一块连续 packed storage；
- 至少两个 expert，形状满足各组相同 N/K；
- 未开启 `ENABLE_ZERO_BUBBLE=1`。

`accumulate` 会映射到 GM6 的 beta：首个 microbatch 使用 beta=0，后续累加使用
beta=1。使用 `main_grad` 时，调用方会把 `grad_added_to_main_grad` 标记为已写入，
避免 Megatron DDP 再加一次梯度。

## 16 机启动示例

以下是已有 16 机、EP8、24 层测试入口的最小形式；hostfile 应替换为当前确认可用的
16 机文件：

```bash
cd /home/jd/haowen.yan/llm_pretrain_script/llm_pretrain_script/cluster
export LOG_NAME=gm6_ws16_$(date +%Y%m%d_%H%M%S)
export TRAINING_STEPS=10
export MATE_GROUPED_GEMM=0
export TE_TN_GM6_WGRAD=1
export SAVE_PATH=/home/jd/haowen.yan/training_runs/outputs/${LOG_NAME}
export LOG_OUTPUT=/home/jd/haowen.yan/training_runs/torchrun_logs

bash auto_fault_manager.sh \
  --hostfile ../hostfile.runtime.jd_llmtest_free_mccl_good16_20260727 \
  --worldsize 16 \
  --dist-run ./dist_run_megatron.sh \
  --output-dir "${SAVE_PATH}" \
  --startup-grace 1800 \
  --hang-minutes 60 \
  --log-error-patterns "RuntimeError,ConnectionError,Segmentation fault,Out of memory,Traceback" \
  --skip-initial-netcheck \
  --daemon
```

如果使用 MATE 路径，把前三个相关变量替换为第二种配置。首次验证建议
`TRAINING_STEPS=2~10`，确认日志中出现 `[TE_TN_GM6]` 或 `mate_tn_gm6_wgrad`，再进行
性能 trace。停止时只停止本次 manager/run，不要误杀其他训练：

```bash
bash auto_fault_manager.sh --hostfile <same-hostfile> --worldsize 16 --stop
```

## 构建和回退

当前 `_tn_gm6.so` 是集群 ABI 的预构建产物。若 Python、PyTorch-MUSA、设备型号或
MUSA/muDNN ABI 改变，需在 MUSA 节点重新构建，不能在普通本地 Windows 环境编译：

```bash
cd megatron-lm-musa-patch/musa_patch/tn_gm6
export TE_TN_GM6_MUDNN_SOURCE_ROOT=/path/to/mudnn
export TE_TN_GM6_DISPATCHER_OBJ=/path/to/dispatcher.o
export TE_TN_GM6_KERNEL_LIB=/path/to/libkernel.so
python setup.py build_ext --inplace
```

生产回退只需取消变量或设为 0：

```bash
export TE_TN_GM6_WGRAD=0
export MATE_TN_GM6_WGRAD=0
```

此时 MATE（若开启）仍可负责 fprop/dgrad，wgrad 回到 Transformer Engine
`general_grouped_gemm`；若同时关闭 MATE，则整个 expert 计算回到原始 TE 路径。

## 排查顺序

1. 先确认每个节点能 `import musa_patch.tn_gm6._tn_gm6`，并确认 Python/PyTorch-MUSA
   ABI 一致。
2. 用 2 step、单一配置验证数值和 `[TE_TN_GM6]` 日志，再扩到 16 机。
3. 若没有 GM6 日志，先检查环境变量是否被 launcher 透传，再检查 layout、dtype、
   packed storage 和空 expert 条件；未命中时是预期的 TE fallback，不等于训练卡死。
4. 性能比较必须使用相同 hostfile、batch、层数、dispatcher、MATE 状态和 profiler
   区间；不要把 profiler 开销与无 profiler 运行直接比较。
