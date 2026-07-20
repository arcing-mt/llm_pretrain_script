# llm_pretrain_script

JD 集群 128 机(1024 卡)MUSA LLM 预训练启动脚本链,取自 pod 内 `/mnt/code/llm_pretrain` 仓库(commit `b4c3470`)。

运行环境:namespace `his-test`,Deployment `jd-llm-pretrain-test`,pod 内工作目录 `/mnt/code/llm_pretrain`。

## 调用链

```
cluster/dist_train_caizhi.sh                       ← 训练入口(daemon 方式拉起)
└─ cluster/auto_fault_manager.sh                   ← 容错守护:起停、hang 检测、故障节点剔除重启
   ├─ cluster/dist_run_megatron.sh                 ← 按 hostfile 逐节点 SSH 分发
   │  └─ scripts/dist_train_megatron_ws128.sh      ← 每节点入口(--worldsize 128 时默认注入)
   │     └─ musa_pretrain_ws128.sh                 ← 实际训练脚本,组装参数后 torchrun
   │        ├─ pretrain_gpt_musa_launcher.py       ← launcher:先 import musa_patch 再 runpy 训练入口
   │        ├─ tokenizer/                          ← HuggingFaceTokenizer(DeepSeek 系)
   │        ├─ /home/Megatron-LM/pretrain_gpt.py   ← 容器内代码(本仓库 Megatron-LM/ 为其拷贝)
   │        └─ /home/megatron-lm-musa-patch/       ← 容器内代码(本仓库 megatron-lm-musa-patch/ 为其拷贝)
   ├─ cluster/hang_detect.sh                       ← 训练 hang 检测(内部调 mccl_bench/stop_all)
   ├─ cluster/mccl_bench.sh                        ← MCCL 通信探测
   └─ cluster/stop_all.sh                          ← 全节点停止
cluster/stop_train_caizhi.sh                       ← 停止入口(fault_manager --stop + stop_all)
hostfile.runtime.128                               ← 128 节点列表
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `cluster/dist_train_caizhi.sh` | 启动入口:`LOG_NAME=ws128_日期`,经 auto_fault_manager 以 `--daemon` 拉起,startup-grace 1800s,hang 检测 60min |
| `cluster/stop_train_caizhi.sh` | 停止入口:fault_manager `--stop`、杀 pid、`stop_all.sh` |
| `cluster/auto_fault_manager.sh` | 容错管理主体(~2100 行):监控日志错误模式、hang 检测、坏节点写 `error_node.txt` 后换节点重启 |
| `cluster/dist_run_megatron.sh` | 分发器:解析 hostfile,选主节点与空闲端口,逐节点 `ssh` 后台执行每节点入口 |
| `cluster/hang_detect.sh` / `cluster/mccl_bench.sh` / `cluster/stop_all.sh` | fault_manager 的探测与停止工具 |
| `scripts/dist_train_megatron_ws128.sh` | 每节点入口(128 机):导出 MUSA/MCCL 环境变量、ulimit,exec `musa_pretrain_ws128.sh` |
| `scripts/dist_train_megatron.sh` | 每节点入口(通用/双机验证,默认 `musa_pretrain_ws2.sh`) |
| `musa_pretrain_ws128.sh` | 128 机正式训练:TP=2+SP、PP=8、EP=64,GBS=16384,seq=4096,61 层 MoE+MLA,对齐 `cuda_pretrain.sh` |
| `musa_pretrain_ws2.sh` | 双机缩小验证版 |
| `pretrain_gpt_musa_launcher.py` | 训练 launcher:强制先 `import musa_patch`,再 runpy `${MCORE_PATH}/pretrain_gpt.py` |
| `hostfile.runtime.128` | 运行时 128 节点 IP 列表 |
| `tokenizer/` | tokenizer 模型与配置(`--tokenizer-type HuggingFaceTokenizer`) |

## 使用方法(pod 内)

```bash
cd /mnt/code/llm_pretrain/cluster

# 启动(每次新 run 会以 ws128_日期 命名,避免 cache/ckpt 冲突)
bash dist_train_caizhi.sh

# 停止
bash stop_train_caizhi.sh
```

关键路径(pod 内):

- 训练输出/ckpt:`/home/jd/wangkang/llm_pretrain/outputs/${LOG_NAME}`
- 数据:`/home/jd/wangkang/llm_pretrain/data/tkn_ds_the_pile`
- Megatron 代码:`/home/Megatron-LM`(首次运行自动 `setup.py build_ext --inplace`)
- MUSA patch:`/home/megatron-lm-musa-patch`
