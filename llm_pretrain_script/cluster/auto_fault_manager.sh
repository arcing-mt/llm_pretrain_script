#!/bin/bash

# Auto launcher + fault detection + log monitor
# Usage:
#   ./auto_fault_manager.sh --hostfile HOSTFILE --worldsize N --logdir LOG_DIR \
#       [--dist-run ./dist_run.sh] [--workload t2v] \
#       [--poll-interval 60] [--stale-min 30] [--disk-threshold 85] \
#       [--error-file ./error_node.txt] [--mccl-bench ./mccl_bench.sh] \
#       [--dingtalk-webhook URL] [--no-start] [--daemon] [--stop]
# Hostfile format: see ../hostfile.test

set -euo pipefail

# 脚本目录和 PID 文件路径
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PID_FILE_BASE="${SCRIPT_DIR}/.auto_fault_manager"
PID_FILE="${PID_FILE_BASE}.pid"
DAEMON_LOG_BASE="${SCRIPT_DIR}/auto_fault_manager.log"
DAEMON_LOG="$DAEMON_LOG_BASE"
DAEMON_MODE=0
SKIP_RUN_PID_CLEANUP=0

# 检查是否已有实例运行
check_single_instance() {
    if [[ -f "$PID_FILE" ]]; then
        local old_pid
        old_pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            log "Another instance is already running (PID: $old_pid), exiting"
            log "Use '$0 --stop' to stop the running instance first"
            exit 0
        else
            # PID 文件存在但进程已不存在，清理旧文件
            rm -f "$PID_FILE"
        fi
    fi
}

# 写入 PID 文件
write_pid_file() {
    echo $$ > "$PID_FILE"
}

# 清理 PID 文件
cleanup_pid_file() {
    rm -f "$PID_FILE"
}

# 通过集群锁 owner 信息停止远端（或本机）正在运行的 manager
# 返回 0 表示已发送停止信号；返回 1 表示未能停止
stop_instance_via_cluster_lock_owner() {
    if [[ -z "${CLUSTER_LOCK_OWNER_FILE:-}" ]] || [[ ! -f "$CLUSTER_LOCK_OWNER_FILE" ]]; then
        return 1
    fi

    local owner_token
    owner_token=$(cat "$CLUSTER_LOCK_OWNER_FILE" 2>/dev/null | head -n 1 | tr -d '\r' || true)
    if [[ -z "$owner_token" ]]; then
        return 1
    fi

    # token 格式: <hostname>-<pid>-<timestamp>
    local owner_pid owner_host
    owner_pid=$(echo "$owner_token" | awk -F- '{print $(NF-1)}')
    owner_host=$(echo "$owner_token" | sed -E 's/-[0-9]+-[0-9]+$//')

    if [[ -z "$owner_host" || -z "$owner_pid" || ! "$owner_pid" =~ ^[0-9]+$ ]]; then
        log "Invalid lock owner token: $owner_token"
        return 1
    fi

    local local_host
    local_host=$(hostname)

    if [[ "$owner_host" == "$local_host" || "$owner_host" == "localhost" || "$owner_host" == "127.0.0.1" ]]; then
        if ps -p "$owner_pid" -o args= 2>/dev/null | grep -F "auto_fault_manager.sh" >/dev/null 2>&1; then
            log "Stopping lock owner instance locally (PID: $owner_pid)..."
            kill -USR1 "$owner_pid" 2>/dev/null || true
            return 0
        fi
        return 1
    fi

    # 远端校验目标进程为 auto_fault_manager 后再发送停止信号
    local ssh_rc=0
    timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 "$owner_host" \
        "ps -p ${owner_pid} -o args= 2>/dev/null | grep -F 'auto_fault_manager.sh' >/dev/null && kill -USR1 ${owner_pid}" \
        >/dev/null 2>&1 || ssh_rc=$?
    if [[ "$ssh_rc" -eq 0 ]]; then
        log "Stop signal sent to lock owner on $owner_host (PID: $owner_pid)"
        return 0
    fi

    log "Failed to stop lock owner on $owner_host (PID: $owner_pid, rc: $ssh_rc)"
    return 1
}

# 停止正在运行的实例
stop_instance() {
    local stopped_any=0

    if [[ -z "${WORLDSIZE:-}" ]]; then
        local pid_files
        pid_files=$(ls -1 "${PID_FILE_BASE}".*.pid 2>/dev/null || true)
        if [[ -z "$pid_files" ]]; then
            log "No running instance found (PID file not exists)"
            exit 0
        fi
        while read -r pid_file; do
            [[ -z "$pid_file" ]] && continue
            local old_pid
            old_pid=$(cat "$pid_file" 2>/dev/null || echo "")
            if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
                log "Stopping running instance (PID: $old_pid, file: $pid_file)..."
                kill -USR1 "$old_pid" 2>/dev/null || true
                local wait_count=0
                while kill -0 "$old_pid" 2>/dev/null && [[ $wait_count -lt 10 ]]; do
                    sleep 1
                    ((wait_count++))
                done
                if kill -0 "$old_pid" 2>/dev/null; then
                    log "Force killing process (PID: $old_pid)..."
                    kill -9 "$old_pid" 2>/dev/null || true
                fi
                rm -f "$pid_file"
                stopped_any=1
            else
                rm -f "$pid_file"
            fi
        done <<< "$pid_files"

        if [[ "$stopped_any" -eq 1 ]]; then
            log "Instance(s) stopped"
        else
            log "No running instance found"
        fi
        exit 0
    fi

    if [[ -f "$PID_FILE" ]]; then
        local old_pid
        old_pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            log "Stopping running instance (PID: $old_pid)..."
            kill -USR1 "$old_pid" 2>/dev/null || true
            # 等待进程退出
            local wait_count=0
            while kill -0 "$old_pid" 2>/dev/null && [[ $wait_count -lt 10 ]]; do
                sleep 1
                ((wait_count++))
            done
            if kill -0 "$old_pid" 2>/dev/null; then
                log "Force killing process..."
                kill -9 "$old_pid" 2>/dev/null || true
            fi
            rm -f "$PID_FILE"
            log "Instance stopped"
        else
            rm -f "$PID_FILE"
            if stop_instance_via_cluster_lock_owner; then
                log "Instance stop requested via cluster lock owner"
            else
                log "No running instance found"
            fi
        fi
    else
        if stop_instance_via_cluster_lock_owner; then
            log "Instance stop requested via cluster lock owner"
        else
            log "No running instance found (PID file not exists)"
        fi
    fi
    exit 0
}

# 以守护进程模式重新启动自身
start_daemon() {
    log "Starting in daemon mode..."
    log "Log file: $DAEMON_LOG"
    log "PID file: $PID_FILE"

    # 移除 --daemon 参数并重新启动
    local args=()
    local skip_next=0
    for arg in "$@"; do
        if [[ $skip_next -eq 1 ]]; then
            skip_next=0
            continue
        fi
        if [[ "$arg" == "--daemon" ]]; then
            continue
        fi
        args+=("$arg")
    done

    # 使用 nohup 在后台运行，重定向输出到日志文件
    nohup bash "$0" "${args[@]}" >> "$DAEMON_LOG" 2>&1 &
    local daemon_pid=$!
    log "Daemon started with PID: $daemon_pid"
    exit 0
}

# 注册退出时的清理函数
cleanup_all() {
    if declare -F stop_hang_detect >/dev/null 2>&1; then
        stop_hang_detect
    fi
    # 终止本地 dist_run 后台进程，避免成为孤儿进程
    if [[ "${SKIP_RUN_PID_CLEANUP:-0}" -ne 1 ]] && [[ "${RUN_PID:-0}" -gt 0 ]] && kill -0 "$RUN_PID" 2>/dev/null; then
        kill "$RUN_PID" 2>/dev/null || true
        wait "$RUN_PID" 2>/dev/null || true
    fi
    if declare -F release_cluster_lock >/dev/null 2>&1; then
        release_cluster_lock
    fi
    cleanup_pid_file
}
trap cleanup_all EXIT
trap 'log "Received termination signal, exiting..."; exit 0' SIGQUIT SIGTERM SIGINT
trap 'log "Received stop-manager signal, exiting without stopping training..."; SKIP_RUN_PID_CLEANUP=1; exit 0' SIGUSR1

# 基础参数与默认值
HOSTFILE=""
WORLDSIZE=""
LOG_DIR=""
OUTPUT_DIR=""
CONFIG_FILE="${SCRIPT_DIR}/auto_fault_manager.conf"
DIST_RUN="${SCRIPT_DIR}/dist_run_fsdp.sh"
HANG_DETECT_SCRIPT="${SCRIPT_DIR}/hang_detect.sh"
WORKLOAD=""
POLL_INTERVAL=60
STALE_MIN=30
HANG_MINUTES=10
HANG_NO_KILL=0
HANG_DETECT_PID=0
DISK_THRESHOLD=85
DISK_PATHS="/,/home/jd"
ERROR_FILE="${SCRIPT_DIR}/error_node.txt"
MCCL_BENCH="${SCRIPT_DIR}/mccl_bench.sh"
STOP_ALL="${SCRIPT_DIR}/stop_all.sh"
DINGTALK_WEBHOOK="${DINGTALK_WEBHOOK:-}"
DINGTALK_AT_ALL=0
WECHAT_WEBHOOK="${WECHAT_WEBHOOK:-}"
WECHAT_MENTIONED_LIST=""
START_TRAIN=1
STOP_ALL_NOW=0
STOP_MANAGERS=0
STOP_SELF=0
# XID 数字白名单（配置文件可覆盖，例如："2000008,2000010"）
DMESG_XID_NUMBERS=""
# dmesg 匹配模式（可通过 --dmesg-xid-pattern(s) 覆盖）
DMESG_XID_PATTERNS=""
# dmesg blocked 进程匹配模式（可通过 --dmesg-blocked-pattern 覆盖）
DMESG_BLOCKED_PATTERN="task .*blocked for more than"
MAX_PARALLEL=32
LOG_ERROR_PATTERNS="RuntimeError,ConnectionError,Segmentation fault,Out of memory,OOM,Traceback"
TRAIN_FINISHED_PATTERN="Training finished"
LOG_NAN_PATTERN="step_loss=nan"
REMOTE_PROC_PATTERN="/usr/bin/python|/usr/local/bin/torchrun|train.py"
RUN_PID=0
CONFIG_MTIME=0
# 训练启动后的宽限期（秒），在此期间不检测进程是否存在
STARTUP_GRACE=120
# 启动宽限期内的就绪节点累计
declare -A GRACE_READY_HOSTS
# 记录训练启动时间
TRAIN_START_TIME=0
# 重启计数（用于区分日志目录）
RESTART_COUNT=0
# 运行日志文件（非 daemon 模式）
RUN_LOG_BASE="${SCRIPT_DIR}/auto_fault_manager_run.log"
RUN_LOG="$RUN_LOG_BASE"
LOG_DIR_BASE=""
CLUSTER_LOCK_PATH="/home/jd/auto_fault_manager.lock"
CLUSTER_LOCK_OWNER_FILE="${CLUSTER_LOCK_PATH}/owner"
CLUSTER_LOCK_HEARTBEAT_FILE="${CLUSTER_LOCK_PATH}/heartbeat"
CLUSTER_LOCK_TOKEN=""
CLUSTER_LOCK_STALE_SEC=180
CLUSTER_LOCK_HEARTBEAT_SEC=30
CLUSTER_LOCK_WAIT_SEC=5
CLUSTER_LOCK_TAKEOVER=0
SKIP_INITIAL_START=0
SKIP_INITIAL_FAULT_DETECT=0
HA_INSTANCES=1
HA_SPAWNED=0
HOSTFILE_SORT_IP=0

# 统一日志输出（同时输出到终端和日志文件）
log() {
    echo "[$(date '+%F %T')] [WS=${WORLDSIZE:-NA}] $*" | tee -a "$RUN_LOG"
}

# 规范化 DingTalk webhook（去除 CR/空白）
normalize_webhook() {
    if [[ -n "${DINGTALK_WEBHOOK:-}" ]]; then
        # 去除 Windows CR 和首尾空白
        DINGTALK_WEBHOOK="${DINGTALK_WEBHOOK//$'\r'/}"
        # trim
        DINGTALK_WEBHOOK="${DINGTALK_WEBHOOK#${DINGTALK_WEBHOOK%%[![:space:]]*}}"
        DINGTALK_WEBHOOK="${DINGTALK_WEBHOOK%${DINGTALK_WEBHOOK##*[![:space:]]}}"
    fi
    if [[ -n "${WECHAT_WEBHOOK:-}" ]]; then
        # 去除 Windows CR 和首尾空白
        WECHAT_WEBHOOK="${WECHAT_WEBHOOK//$'\r'/}"
        # trim
        WECHAT_WEBHOOK="${WECHAT_WEBHOOK#${WECHAT_WEBHOOK%%[![:space:]]*}}"
        WECHAT_WEBHOOK="${WECHAT_WEBHOOK%${WECHAT_WEBHOOK##*[![:space:]]}}"
    fi
}

# 参数用法说明
usage() {
    cat <<EOF
Usage: $0 --hostfile HOSTFILE --worldsize N [--logdir LOG_DIR] [--output-dir OUTPUT_DIR] [options]
Hostfile format: see ../hostfile.test

Options:
    --config PATH           Optional config file (default: ./auto_fault_manager.conf)
    --dist-run PATH         Path to dist_run.sh (default: ./dist_run.sh)
        --stop-all PATH         Path to stop_all.sh (default: ./stop_all.sh)
  --logdir PATH           Log directory (default: auto-generated with timestamp)
    --output-dir PATH       Output directory passed to dist_run (default: same as logdir)
  --workload NAME         Workload argument for dist_run.sh (optional)
  --poll-interval SEC     Poll interval in seconds (default: 60)
  --stale-min MIN         Stale threshold in minutes (default: 30)
    --disk-threshold PCT    Disk usage threshold percent (default: 85)
    --disk-paths CSV        Disk paths to check (default: /,/home/jd)
    --cluster-lock-path PATH Shared lock directory path (default: /home/jd/auto_fault_manager.lock)
    --cluster-lock-stale-sec SEC Stale threshold for cluster lock heartbeat (default: 180)
    --cluster-lock-heartbeat-sec SEC Heartbeat interval seconds (default: 30)
    --cluster-lock-wait-sec SEC Standby wait seconds when lock held (default: 5)
    --error-file PATH       Error node file (default: ./error_node.txt)
    --mccl-bench PATH       Path to mccl_bench.sh (default: ./mccl_bench.sh)
    --hang-detect-script PATH Path to hang_detect.sh (default: ./hang_detect.sh)
    --hang-minutes MIN      Hang threshold in minutes since last step/loss log (default: 10)
    --hang-no-kill          Detect hang and dump stack only, do not stop/restart training
  --dingtalk-webhook URL  DingTalk webhook URL (or env DINGTALK_WEBHOOK)
    --dingtalk-at-all       @all in DingTalk alerts (default: off)
  --wechat-webhook URL    WeChat Work webhook URL (or env WECHAT_WEBHOOK)
    --wechat-mentioned-list CSV Comma-separated user IDs to mention (or @all)
    --hostfile-sort-ip      Sort hostfile entries by IP before use (default: off)
  --no-start              Do not start dist_run.sh automatically
        --stop-all-now          Stop all processes via stop_all.sh and exit
        --stop-managers         Stop auto_fault_manager on all hosts in hostfile and exit
    --daemon                Run in daemon mode (background, survives terminal close)
    --stop                  Stop the running instance
    --skip-initial-netcheck Skip mccl netcheck during initial fault detection
    --netcheck-timeout SEC  Timeout for mccl netcheck in seconds (default: 1800)
    --dmesg-xid-pattern PAT Pattern for dmesg XID check (regex supported)
    --dmesg-xid-patterns CSV Comma-separated patterns (regex supported)
    --dmesg-xid-numbers CSV Comma-separated XID numbers to match (e.g., 2000008,2000010)
    --dmesg-blocked-pattern REGEX Pattern for dmesg blocked task check (regex supported)
        --log-error-patterns CSV Comma-separated log error patterns (regex supported)
        --remote-proc-pattern REGEX Remote train process regex for pgrep (default: /usr/bin/python|/usr/local/bin/torchrun|train.py)
    --startup-grace SEC     Grace period after train start before process check (default: 120)
    --ha-instances N        Start N HA instances across hosts (default: 1)
    --ha-spawned            Internal flag to avoid recursive HA spawn
EOF
}

# 获取文件修改时间（兼容不同 stat 实现）
get_mtime() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo 0
        return
    fi
    if stat -c %Y "$file" >/dev/null 2>&1; then
        stat -c %Y "$file"
    else
        stat -f %m "$file" 2>/dev/null || echo 0
    fi
}

# 加载配置文件（如存在）
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$CONFIG_FILE"
        normalize_webhook
    fi
}

# 配置变更检测并热加载
reload_config_if_needed() {
    local mtime
    mtime=$(get_mtime "$CONFIG_FILE")
    if [[ "$mtime" -ne "$CONFIG_MTIME" ]]; then
        CONFIG_MTIME="$mtime"
        load_config
        log "Config reloaded: $CONFIG_FILE"
    fi
}

# 控制并发 SSH 任务数
wait_for_slot() {
    local max="${1:-$MAX_PARALLEL}"
    while true; do
        local running
        running=$(jobs -pr 2>/dev/null | wc -l | tr -d ' ' || true)
        running=${running:-0}
        if [[ "$running" -lt "$max" ]]; then
            break
        fi
        sleep 0.2
    done
}

# 解析命令行参数
# 保存原始参数用于 daemon 模式重启
ORIGINAL_ARGS=("$@")

# 先预扫描 --config 参数
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    if [[ "$arg" == "--config" ]]; then
        next=$((i+1))
        CONFIG_FILE="${!next}"
        break
    fi
done

# 加载配置文件（如存在），作为默认值
if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

# 命令行参数解析（会覆盖配置文件的值）
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="$2"; shift 2;;
        --hostfile) HOSTFILE="$2"; shift 2;;
        --worldsize) WORLDSIZE="$2"; shift 2;;
        --logdir) LOG_DIR="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --dist-run) DIST_RUN="$2"; shift 2;;
        --stop-all) STOP_ALL="$2"; shift 2;;
        --workload) WORKLOAD="$2"; shift 2;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2;;
        --stale-min) STALE_MIN="$2"; shift 2;;
        --disk-threshold) DISK_THRESHOLD="$2"; shift 2;;
        --disk-paths) DISK_PATHS="$2"; shift 2;;
        --cluster-lock-path) CLUSTER_LOCK_PATH="$2"; shift 2;;
        --cluster-lock-stale-sec) CLUSTER_LOCK_STALE_SEC="$2"; shift 2;;
        --cluster-lock-heartbeat-sec) CLUSTER_LOCK_HEARTBEAT_SEC="$2"; shift 2;;
        --cluster-lock-wait-sec) CLUSTER_LOCK_WAIT_SEC="$2"; shift 2;;
        --error-file) ERROR_FILE="$2"; shift 2;;
        --mccl-bench) MCCL_BENCH="$2"; shift 2;;
        --hang-detect-script) HANG_DETECT_SCRIPT="$2"; shift 2;;
        --hang-minutes) HANG_MINUTES="$2"; shift 2;;
        --hang-no-kill) HANG_NO_KILL=1; shift;;
        --dingtalk-webhook) DINGTALK_WEBHOOK="$2"; shift 2;;
        --wechat-webhook) WECHAT_WEBHOOK="$2"; shift 2;;
        --wechat-mentioned-list) WECHAT_MENTIONED_LIST="$2"; shift 2;;
        --dingtalk-at-all) DINGTALK_AT_ALL=1; shift;;
        --dmesg-xid-pattern) DMESG_XID_PATTERNS="$2"; shift 2;;
        --dmesg-xid-patterns) DMESG_XID_PATTERNS="$2"; shift 2;;
        --dmesg-xid-numbers) DMESG_XID_NUMBERS="$2"; shift 2;;
        --dmesg-blocked-pattern) DMESG_BLOCKED_PATTERN="$2"; shift 2;;
        --log-error-patterns) LOG_ERROR_PATTERNS="$2"; shift 2;;
        --remote-proc-pattern) REMOTE_PROC_PATTERN="$2"; shift 2;;
        --skip-initial-netcheck) SKIP_INITIAL_NETCHECK=1; shift;;
        --netcheck-timeout) NETCHECK_TIMEOUT="$2"; shift 2;;
        --startup-grace) STARTUP_GRACE="$2"; shift 2;;
        --ha-instances) HA_INSTANCES="$2"; shift 2;;
        --ha-spawned) HA_SPAWNED=1; shift;;
        --hostfile-sort-ip) HOSTFILE_SORT_IP=1; shift;;
        --no-start) START_TRAIN=0; shift;;
        --stop-all-now) STOP_ALL_NOW=1; shift;;
        --stop-managers) STOP_MANAGERS=1; shift;;
        --daemon) DAEMON_MODE=1; shift;;
        --stop) STOP_SELF=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1"; usage; exit 1;;
    esac
done

# 规范化 webhook（命令行参数覆盖后）
normalize_webhook

# 设置运行日志文件名（包含 worldsize 与时间）
if [[ -n "${WORLDSIZE:-}" ]]; then
    CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
    RUN_LOG="${SCRIPT_DIR}/auto_fault_manager_run_ws${WORLDSIZE}_${CURRENT_TIME}.log"
    DAEMON_LOG="${SCRIPT_DIR}/auto_fault_manager_ws${WORLDSIZE}_${CURRENT_TIME}.log"
else
    CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
    RUN_LOG="${SCRIPT_DIR}/auto_fault_manager_run_${CURRENT_TIME}.log"
    DAEMON_LOG="${SCRIPT_DIR}/auto_fault_manager_${CURRENT_TIME}.log"
fi

# 如果是 daemon 模式，先以后台方式重新启动自身
if [[ "$DAEMON_MODE" -eq 1 ]]; then
    start_daemon "${ORIGINAL_ARGS[@]}"
fi

# 将相对路径转换为绝对路径（基于脚本目录）
resolve_path() {
    local path="$1"
    if [[ "$path" != /* ]]; then
        # 相对路径，转换为基于脚本目录的绝对路径
        path="${SCRIPT_DIR}/${path#./}"
    fi
    echo "$path"
}

# 启动 HA 多实例（在不同主机上启动同一脚本）
start_ha_instances() {
    if [[ "$HA_INSTANCES" -le 1 || "$HA_SPAWNED" -eq 1 ]]; then
        return
    fi
    if [[ -z "$HOSTFILE" ]]; then
        return
    fi
    local targets=()
    while read -r line; do
        local host
        host=$(get_host_from_line "$line")
        [[ -z "$host" ]] && continue
        if [[ "$host" == "$(hostname)" || "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
            continue
        fi
        targets+=("$host")
        if [[ ${#targets[@]} -ge $((HA_INSTANCES - 1)) ]]; then
            break
        fi
    done < <(read_host_lines)

    if [[ ${#targets[@]} -eq 0 ]]; then
        return
    fi

    log "Starting HA instances: total=${HA_INSTANCES}, remotes=${#targets[@]} (${targets[*]})"
    local args_escaped=()
    for arg in "${ORIGINAL_ARGS[@]}"; do
        args_escaped+=("$(printf '%q' "$arg")")
    done
    local args_str="${args_escaped[*]}"
    for host in "${targets[@]}"; do
        (
            local ssh_out
            ssh_out=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "bash '${SCRIPT_DIR}/auto_fault_manager.sh' ${args_str} --ha-spawned --daemon" 2>&1)
            local ssh_rc=$?
            if [[ "$ssh_rc" -ne 0 ]]; then
                log "Warning: failed to start HA instance on $host (rc: $ssh_rc)"
                if [[ -n "$ssh_out" ]]; then
                    log "Warning: ssh output on $host: $ssh_out"
                fi
            fi
        ) &
        wait_for_slot
    done
    wait
}

# 去除日志目录末尾的重启后缀（如 _r1_r2）
strip_restart_suffix() {
    echo "$1" | sed -E 's/(_r[0-9]+)+$//'
}

# 生成实例标签（用于多任务区分）
build_instance_tag() {
    local host_tag=""
    if [[ -n "${HOSTFILE:-}" ]]; then
        host_tag=$(basename "$HOSTFILE")
        host_tag=$(echo "$host_tag" | sed -E 's/[^a-zA-Z0-9]+/_/g')
    fi
    local ws_tag="${WORLDSIZE:-NA}"
    if [[ -n "$host_tag" ]]; then
        echo "${host_tag}_ws${ws_tag}"
    else
        echo "ws${ws_tag}"
    fi
}

# 读取有效 host 行（忽略注释与空行，格式参考 ../hostfile.test）
read_host_lines() {
    local lines
    lines=$(grep -v '^#' "$HOSTFILE" | awk 'NF {print}')
    if [[ "${HOSTFILE_SORT_IP:-0}" -eq 1 ]]; then
        echo "$lines" | sort -t . -k1,1n -k2,2n -k3,3n -k4,4n
    else
        echo "$lines"
    fi
}

# 从 hostfile 行中提取主机（格式示例："10.121.3.1 slots=8"）
get_host_from_line() {
    echo "$1" | awk '{print $1}'
}

# 基于 mthreads-gmi 输出判断每张卡是否都有显存 > threshold 的进程
# 返回 0 表示所有 GPU 均满足；返回 1 表示不满足或无法判断
gmi_all_gpus_over_threshold() {
    local gmi_output="$1"
    local threshold="${2:-500}"

    if [[ -z "$gmi_output" ]]; then
        return 1
    fi

    # 先解析 GPU 汇总区（Processes 之前），取每张卡的显存占用
    local summary_gpu_count=0
    local summary_ready_count=0
    local summary_lines
    summary_lines=$(echo "$gmi_output" | awk '/^Processes:/ {exit} /^[0-9]+[[:space:]]+/{print}')
    if [[ -n "$summary_lines" ]]; then
        while read -r line; do
            [[ -z "$line" ]] && continue
            local used
            used=$(echo "$line" | grep -oE '[0-9]+MiB\([0-9]+MiB\)' | head -n 1 | sed -E 's/MiB.*//')
            if [[ -n "$used" ]]; then
                summary_gpu_count=$((summary_gpu_count + 1))
                if [[ "$used" -gt "$threshold" ]]; then
                    summary_ready_count=$((summary_ready_count + 1))
                fi
            fi
        done <<< "$summary_lines"
        if [[ "$summary_gpu_count" -gt 0 ]]; then
            if [[ "$summary_ready_count" -eq "$summary_gpu_count" ]]; then
                return 0
            fi
            return 1
        fi
    fi

    # 汇总区无法解析时，回退到 Processes 区（逐卡取最大显存进程）
    local proc_lines
    proc_lines=$(echo "$gmi_output" | awk '/^Processes:/ {flag=1; next} flag && /^[0-9]+[[:space:]]+[0-9]+/{print}')
    if [[ -z "$proc_lines" ]]; then
        return 1
    fi

    declare -A gpu_max
    while read -r line; do
        [[ -z "$line" ]] && continue
        local gpu_id used
        gpu_id=$(echo "$line" | awk '{print $1}')
        used=$(echo "$line" | grep -oE '[0-9]+MiB$' | sed 's/MiB//')
        if [[ -n "$gpu_id" && -n "$used" ]]; then
            if [[ -z "${gpu_max[$gpu_id]+x}" || "$used" -gt "${gpu_max[$gpu_id]}" ]]; then
                gpu_max[$gpu_id]="$used"
            fi
        fi
    done <<< "$proc_lines"

    if [[ ${#gpu_max[@]} -eq 0 ]]; then
        return 1
    fi

    local ok=0
    for k in "${!gpu_max[@]}"; do
        if [[ "${gpu_max[$k]}" -gt "$threshold" ]]; then
            ok=$((ok + 1))
        fi
    done
    if [[ "$ok" -eq "${#gpu_max[@]}" ]]; then
        return 0
    fi
    return 1
}

# 读取已有故障节点列表
load_error_nodes() {
    if [[ -f "$ERROR_FILE" ]]; then
        awk 'NF {print $1}' "$ERROR_FILE" | sort -u
    fi
}

# 记录故障节点及原因（去重）
add_error_node() {
    local node="$1"
    local reason="$2"
    if [[ -z "$node" ]]; then
        return
    fi
    if [[ -f "$ERROR_FILE" ]] && grep -qE "^${node}(\s|$)" "$ERROR_FILE"; then
        return
    fi
    printf '%s\t%s\t%s\n' "$node" "$reason" "$(date '+%F %T')" >> "$ERROR_FILE"
}

# 停止所有节点上的 auto_fault_manager 进程
stop_auto_fault_managers_cluster() {
    if [[ ! -f "$HOSTFILE" ]]; then
        log "Error: hostfile not found: $HOSTFILE"
        exit 1
    fi
    log "Stopping auto_fault_manager on all hosts in $HOSTFILE"
    while read -r line; do
        local host
        host=$(get_host_from_line "$line")
        (
            if [[ -z "$host" ]]; then
                exit 0
            fi
            if [[ "$host" == "$(hostname)" || "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
                stop_instance
            else
                local ssh_out
                ssh_out=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "$host" "bash '${SCRIPT_DIR}/auto_fault_manager.sh' --hostfile '${HOSTFILE}' --worldsize '${WORLDSIZE}' --stop" 2>&1)
                local ssh_rc=$?
                if [[ "$ssh_rc" -ne 0 ]]; then
                    log "Warning: failed to stop on $host (ssh rc: $ssh_rc)"
                    if [[ -n "$ssh_out" ]]; then
                        log "Warning: ssh output on $host: $ssh_out"
                    fi
                fi
            fi
        ) &
        wait_for_slot
    done < <(grep -v '^#' "$HOSTFILE" | awk 'NF {print}')
    wait
    log "Stop-managers requested, done"
    exit 0
}

# 集群级别锁，确保只运行一个 auto_fault_manager（支持 HA 接管）
write_cluster_lock_owner() {
    printf '%s\n' "$CLUSTER_LOCK_TOKEN" > "$CLUSTER_LOCK_OWNER_FILE" || true
}

heartbeat_cluster_lock() {
    if [[ -z "$CLUSTER_LOCK_TOKEN" ]]; then
        return
    fi
    if [[ -f "$CLUSTER_LOCK_OWNER_FILE" ]] && grep -qx "$CLUSTER_LOCK_TOKEN" "$CLUSTER_LOCK_OWNER_FILE"; then
        date +%s > "$CLUSTER_LOCK_HEARTBEAT_FILE" || true
    fi
}

get_cluster_lock_last_ts() {
    if [[ -f "$CLUSTER_LOCK_HEARTBEAT_FILE" ]]; then
        local ts
        ts=$(cat "$CLUSTER_LOCK_HEARTBEAT_FILE" 2>/dev/null || echo 0)
        if [[ "$ts" =~ ^[0-9]+$ ]]; then
            echo "$ts"
            return
        fi
    fi
    if [[ -d "$CLUSTER_LOCK_PATH" ]]; then
        get_mtime "$CLUSTER_LOCK_PATH"
        return
    fi
    echo 0
}

is_cluster_lock_stale() {
    local last_ts
    last_ts=$(get_cluster_lock_last_ts)
    if [[ "$last_ts" -eq 0 ]]; then
        return 0
    fi
    local now
    now=$(date +%s)
    if [[ $((now - last_ts)) -gt "$CLUSTER_LOCK_STALE_SEC" ]]; then
        return 0
    fi
    return 1
}

prompt_cleanup_resources() {
    local reason="$1"
    if [[ "$DAEMON_MODE" -eq 1 || ! -t 0 ]]; then
        return 1
    fi
    local ans
    read -r -p "Stale lock detected (${reason}). Clean resources for current tag only and continue? [y/N]: " ans
    case "$ans" in
        y|Y|yes|YES) return 0;;
        *) return 1;;
    esac
}

acquire_cluster_lock() {
    CLUSTER_LOCK_TOKEN="$(hostname)-$$-$(date +%s)"
    CLUSTER_LOCK_TAKEOVER=0
    while true; do
        if mkdir -p "$CLUSTER_LOCK_PATH" 2>/dev/null; then
            write_cluster_lock_owner
            heartbeat_cluster_lock
            log "Cluster lock acquired: $CLUSTER_LOCK_PATH"
            return 0
        fi

        local owner_msg=""
        if [[ -f "$CLUSTER_LOCK_OWNER_FILE" ]]; then
            owner_msg="owner: $(cat "$CLUSTER_LOCK_OWNER_FILE" 2>/dev/null || echo unknown)"
        fi

        if is_cluster_lock_stale; then
            local last_ts now age
            last_ts=$(get_cluster_lock_last_ts)
            now=$(date +%s)
            if [[ "$last_ts" =~ ^[0-9]+$ && "$last_ts" -gt 0 ]]; then
                age=$((now - last_ts))
            else
                age=-1
            fi
            local has_owner=0
            local has_heartbeat=0
            if [[ -f "$CLUSTER_LOCK_OWNER_FILE" ]]; then
                has_owner=1
            fi
            if [[ -f "$CLUSTER_LOCK_HEARTBEAT_FILE" ]]; then
                has_heartbeat=1
            fi

            log "Cluster lock stale, attempting takeover ($CLUSTER_LOCK_PATH ${owner_msg}, age=${age}s)"
            if [[ "$has_owner" -eq 0 && "$has_heartbeat" -eq 0 ]]; then
                log "Orphan lock detected (no owner/heartbeat), auto cleanup"
            else
                if prompt_cleanup_resources "lock=${CLUSTER_LOCK_PATH}, age=${age}s"; then
                    log "User approved cleanup, removing stale lock dir"
                else
                    if [[ "$HA_INSTANCES" -gt 1 || "$HA_SPAWNED" -eq 1 ]]; then
                        log "HA mode: auto cleanup stale lock (age=${age}s)"
                    else
                        log "Cleanup declined or not interactive; exiting"
                        exit 1
                    fi
                fi
            fi
            rm -rf "$CLUSTER_LOCK_PATH" >/dev/null 2>&1 || true
            if [[ -d "$CLUSTER_LOCK_PATH" ]]; then
                log "Error: failed to remove stale lock dir: $CLUSTER_LOCK_PATH"
                log "Error: please remove it manually if no instance is running"
                exit 1
            fi
            sleep 1
            if mkdir -p "$CLUSTER_LOCK_PATH" 2>/dev/null; then
                CLUSTER_LOCK_TAKEOVER=1
                write_cluster_lock_owner
                heartbeat_cluster_lock
                log "Cluster lock takeover successful: $CLUSTER_LOCK_PATH"
                return 0
            else
                log "Error: failed to create lock dir after cleanup: $CLUSTER_LOCK_PATH"
                log "Error: check permissions or parent path, then retry"
                exit 1
            fi
        else
            log "Cluster lock held, entering standby ($CLUSTER_LOCK_PATH ${owner_msg})"
        fi
        sleep "$CLUSTER_LOCK_WAIT_SEC"
    done
}

release_cluster_lock() {
    if [[ -z "$CLUSTER_LOCK_TOKEN" ]]; then
        return
    fi
    if [[ -f "$CLUSTER_LOCK_OWNER_FILE" ]] && grep -qx "$CLUSTER_LOCK_TOKEN" "$CLUSTER_LOCK_OWNER_FILE"; then
        rm -rf "$CLUSTER_LOCK_PATH" >/dev/null 2>&1 || true
    fi
}

# 检测远端训练进程是否异常退出（通过GPU显存占用检查）
# 返回 0 表示有进程缺失，返回 1 表示所有进程正常
# 注意：不在此函数中添加 error_node，由调用方根据上下文决定是否标记故障
check_process_exit() {
    local hosts_file="$RUN_HOSTFILE"
    if [[ ! -f "$hosts_file" ]]; then
        hosts_file="$HOSTFILE"
    fi
    local first_line
    first_line=$(grep -v '^#' "$hosts_file" | awk 'NF {print; exit}')
    if [[ -z "$first_line" ]]; then
        log "Process check: no hosts found in $hosts_file"
        return 1
    fi

    local host
    host=$(get_host_from_line "$first_line")

    local has_process=0
    local gmi_output
    gmi_output=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "mthreads-gmi 2>/dev/null" 2>/dev/null || true)

    if gmi_all_gpus_over_threshold "$gmi_output" 500; then
        has_process=1
    fi

    if [[ "$has_process" -eq 0 ]]; then
        log "Process check: FAULT - $host GPU mem < 500MB"
        return 0
    fi

    log "Process check: ok"
    return 1
}

# 获取训练进程就绪主机列表（每行一个 IP 输出到 stdout）
# 全量检测所有节点，每 32 个 IP 为一组，组间 sleep 1 秒
# 注意：本函数通过 $() 调用，log 必须重定向到 >&2 避免污染 stdout
get_process_ready_hosts() {
    local hosts_file="$RUN_HOSTFILE"
    if [[ ! -f "$hosts_file" ]]; then
        hosts_file="$HOSTFILE"
    fi

    mapfile -t _host_lines < <(grep -v '^#' "$hosts_file" | awk 'NF {print}')
    local total=${#_host_lines[@]}
    if [[ "$total" -eq 0 ]]; then
        log "Startup grace ready nodes: 0/0 (no hosts)" >&2
        return
    fi

    local ready_file
    ready_file=$(mktemp)

    local group_size=32
    local i=0
    while [[ "$i" -lt "$total" ]]; do
        local pids=()
        local end=$((i + group_size))
        if [[ "$end" -gt "$total" ]]; then
            end=$total
        fi
        for ((j=i; j<end; j++)); do
            local line host
            line="${_host_lines[$j]}"
            host=$(get_host_from_line "$line")
            (
                local has_process=0
                local gmi_output
                gmi_output=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "mthreads-gmi 2>/dev/null" 2>/dev/null || true)

                if gmi_all_gpus_over_threshold "$gmi_output" 500; then
                    has_process=1
                fi

                if [[ "$has_process" -eq 1 ]]; then
                    echo "$host" >> "$ready_file"
                fi
            ) &
            pids+=($!)
            wait_for_slot
        done

        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        i=$end
        if [[ "$i" -lt "$total" ]]; then
            sleep 1
        fi
    done

    local ready_count=0
    if [[ -s "$ready_file" ]]; then
        sort -u "$ready_file"
        ready_count=$(wc -l < "$ready_file" | tr -d ' ')
    fi
    rm -f "$ready_file"

    log "Startup grace ready nodes: ${ready_count}/${total}" >&2
}

reset_grace_ready_hosts() {
    GRACE_READY_HOSTS=()
}

determine_takeover_mode() {
    if [[ "$CLUSTER_LOCK_TAKEOVER" -ne 1 ]]; then
        return
    fi
    log "Takeover mode detected, checking existing training process state..."
    if build_run_hostfile; then
        if ! check_process_exit; then
            SKIP_INITIAL_START=1
            SKIP_INITIAL_FAULT_DETECT=1
            TRAIN_START_TIME=$(date +%s)
            log "Training appears running, skip initial start and enter monitor"
            return
        fi
    fi
    log "Training not healthy or not running, will start from scratch"
}

# 生成剔除故障节点后的运行 hostfile
build_run_hostfile() {
    local tmp_file
    tmp_file=$(mktemp)
    local error_nodes
    error_nodes=$(load_error_nodes || true)

    if [[ -n "$error_nodes" ]]; then
        awk 'NR==FNR {bad[$1]=1; next} !($1 in bad)' <(echo "$error_nodes") <(read_host_lines) > "$tmp_file"
    else
        read_host_lines > "$tmp_file"
    fi

    local count
    count=$(wc -l < "$tmp_file" | tr -d ' ')
    if [[ "$count" -lt "$WORLDSIZE" ]]; then
        log "Error: available hosts ($count) less than worldsize ($WORLDSIZE)"
        rm -f "$tmp_file"
        return 1
    fi

    sort -t . -k1,1n -k2,2n -k3,3n -k4,4n "$tmp_file" | head -n "$WORLDSIZE" > "$RUN_HOSTFILE"
    rm -f "$tmp_file"
    log "Generated $RUN_HOSTFILE with $WORLDSIZE hosts"
}



# 停止训练（stop_all.sh）
stop_train() {
    # 先终止本地 dist_run 后台进程，避免残留进程干扰
    if [[ "${RUN_PID:-0}" -gt 0 ]] && kill -0 "$RUN_PID" 2>/dev/null; then
        log "Killing local dist_run process (PID: $RUN_PID)..."
        kill "$RUN_PID" 2>/dev/null || true
        wait "$RUN_PID" 2>/dev/null || true
        RUN_PID=0
    fi
    if [[ ! -f "$STOP_ALL" ]]; then
        log "stop_all.sh not found: $STOP_ALL"
        return 1
    fi
    log "Stopping existing processes via $STOP_ALL $HOSTFILE $MAX_PARALLEL"
    local stop_rc=0
    timeout 120 bash "$STOP_ALL" "$HOSTFILE" "$MAX_PARALLEL" || stop_rc=$?
    if [[ "$stop_rc" -eq 124 ]]; then
        log "Warning: stop_all.sh timed out after 120s, some nodes may not have been cleaned"
    fi
}

# 解析路径参数
DIST_RUN=$(resolve_path "$DIST_RUN")
STOP_ALL=$(resolve_path "$STOP_ALL")
MCCL_BENCH=$(resolve_path "$MCCL_BENCH")
HANG_DETECT_SCRIPT=$(resolve_path "$HANG_DETECT_SCRIPT")
ERROR_FILE=$(resolve_path "$ERROR_FILE")
CLUSTER_LOCK_PATH=$(resolve_path "$CLUSTER_LOCK_PATH")
CLUSTER_LOCK_OWNER_FILE="${CLUSTER_LOCK_PATH}/owner"
CLUSTER_LOCK_HEARTBEAT_FILE="${CLUSTER_LOCK_PATH}/heartbeat"
if [[ -n "$HOSTFILE" && "$HOSTFILE" != /* ]]; then
    HOSTFILE=$(resolve_path "$HOSTFILE")
fi
if [[ -n "$LOG_DIR" && "$LOG_DIR" != /* ]]; then
    LOG_DIR=$(resolve_path "$LOG_DIR")
fi
if [[ -n "$OUTPUT_DIR" && "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR=$(resolve_path "$OUTPUT_DIR")
fi

# 基于 hostfile/worldsize 生成实例标签，避免多任务冲突
INSTANCE_TAG=$(build_instance_tag)
PID_FILE="${PID_FILE_BASE}.${INSTANCE_TAG}.pid"
CLUSTER_LOCK_PATH="${CLUSTER_LOCK_PATH}/${INSTANCE_TAG}"
CLUSTER_LOCK_OWNER_FILE="${CLUSTER_LOCK_PATH}/owner"
CLUSTER_LOCK_HEARTBEAT_FILE="${CLUSTER_LOCK_PATH}/heartbeat"

# 仅停止本机实例并退出
if [[ "$STOP_SELF" -eq 1 ]]; then
    stop_instance
fi

# 检查单实例运行（按实例标签）
check_single_instance

# 写入 PID 文件
write_pid_file

CONFIG_MTIME=$(get_mtime "$CONFIG_FILE")

# 校验必需参数
if [[ "$STOP_MANAGERS" -eq 1 || "$STOP_ALL_NOW" -eq 1 ]]; then
    if [[ -z "$HOSTFILE" ]]; then
        usage
        exit 1
    fi
else
    if [[ -z "$HOSTFILE" || -z "$WORLDSIZE" ]]; then
        usage
        exit 1
    fi
fi

# 如果未指定 LOG_DIR，自动生成默认目录
if [[ -z "$LOG_DIR" ]]; then
    CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
    LOG_DIR="${SCRIPT_DIR}/logs_ws${WORLDSIZE}_${CURRENT_TIME}"
    log "LOG_DIR not specified, using default: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

# 如果未指定 OUTPUT_DIR，默认与 LOG_DIR 一致
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$LOG_DIR"
fi

# 记录基础日志目录，避免重启时不断追加 _rN
LOG_DIR_BASE=$(strip_restart_suffix "$LOG_DIR")

# 记录启动信息
log "=========================================="
log "Auto Fault Manager Started"
log "Hostfile: $HOSTFILE"
log "Worldsize: $WORLDSIZE"
log "Log directory: $LOG_DIR"
log "Output directory: $OUTPUT_DIR"
log "Run log: $RUN_LOG"
log "=========================================="

# 校验 hostfile 存在
if [[ ! -f "$HOSTFILE" ]]; then
    log "Error: hostfile not found: $HOSTFILE"
    exit 1
fi

# 启动 HA 多实例（仅主实例触发）
start_ha_instances

# 仅执行 stop_all 并退出
if [[ "$STOP_ALL_NOW" -eq 1 ]]; then
    log "Stop-all requested, invoking $STOP_ALL"
    stop_train || true
    exit 0
fi

# 仅停止所有 auto_fault_manager 并退出
if [[ "$STOP_MANAGERS" -eq 1 ]]; then
    stop_auto_fault_managers_cluster
fi

# 校验 worldsize 合法
if ! [[ "$WORLDSIZE" =~ ^[0-9]+$ ]] || [[ "$WORLDSIZE" -lt 1 ]]; then
    log "Error: worldsize must be a positive integer"
    exit 1
fi

# 128 机正式交付默认入口（无需再手工 export；试验脚本仍可用 env 覆盖）
if [[ "$WORLDSIZE" -ge 128 ]]; then
    export LLM_PRETRAIN_DIST_TRAIN="${LLM_PRETRAIN_DIST_TRAIN:-scripts/dist_train_megatron_ws128.sh}"
    export MUSA_PRETRAIN_ENTRY="${MUSA_PRETRAIN_ENTRY:-musa_pretrain_ws128.sh}"
    log "ws128 defaults: LLM_PRETRAIN_DIST_TRAIN=${LLM_PRETRAIN_DIST_TRAIN} MUSA_PRETRAIN_ENTRY=${MUSA_PRETRAIN_ENTRY}"
fi

# 获取集群锁，确保只有一个 auto_fault_manager 生效
acquire_cluster_lock

# 运行时使用的 hostfile
RUN_HOSTFILE="${HOSTFILE}.run.${WORLDSIZE}"

# 接管场景下判定是否跳过初始启动
determine_takeover_mode

# musaInfo 日志目录
MUSAINFO_LOG_DIR="${SCRIPT_DIR}/musainfo_logs"

# 发送钉钉告警（若配置了 webhook）
send_dingtalk() {
    local msg="$1"
    if [[ -z "$DINGTALK_WEBHOOK" ]]; then
        log "DingTalk webhook not set, skip alert: $msg"
        return
    fi
    if ! command -v curl >/dev/null 2>&1; then
        log "DingTalk alert failed: curl not found"
        return
    fi
    local curl_rc=0
    local at_all
    at_all=$(echo "${DINGTALK_AT_ALL:-0}" | tr -d '"' | tr -d "'")
    local is_at_all=false
    if [[ "$at_all" == "1" || "$at_all" == "true" ]]; then
        is_at_all=true
    fi
    # 使用 printf 进行 JSON 转义
    local escaped_msg
    escaped_msg=$(printf '%s' "$msg" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' 2>/dev/null | sed 's/^"//;s/"$//' || printf '%s' "$msg" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g' | tr '\n' ' ')
    local json_data
    json_data="{\"msgtype\":\"text\",\"text\":{\"content\":\"${escaped_msg}\"},\"at\":{\"isAtAll\":${is_at_all}}}"
    log "DingTalk sending: $json_data"
    local response
    response=$(curl -s -X POST "$DINGTALK_WEBHOOK" \
        -H 'Content-Type: application/json' \
        -d "$json_data" 2>&1) || curl_rc=$?
    if [[ "$curl_rc" -ne 0 ]]; then
        log "DingTalk alert failed (curl exit: $curl_rc)"
    else
        log "DingTalk response: $response"
        log "DingTalk alert sent"
    fi
}

# 发送企业微信告警（若配置了 webhook）
send_wechat() {
    local msg="$1"
    if [[ -z "$WECHAT_WEBHOOK" ]]; then
        log "WeChat webhook not set, skip alert: $msg"
        return
    fi
    if ! command -v curl >/dev/null 2>&1; then
        log "WeChat alert failed: curl not found"
        return
    fi
    local curl_rc=0
    # 使用 printf 进行 JSON 转义
    local escaped_msg
    escaped_msg=$(printf '%s' "$msg" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' 2>/dev/null | sed 's/^"//;s/"$//' || printf '%s' "$msg" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g' | tr '\n' ' ')

    # 构建企业微信消息JSON
    local json_data
    if [[ -n "$WECHAT_MENTIONED_LIST" ]]; then
        # 处理 mentioned_list，支持 @all 或用户ID列表
        if [[ "$WECHAT_MENTIONED_LIST" == "@all" ]]; then
            json_data="{\"msgtype\":\"text\",\"text\":{\"content\":\"${escaped_msg}\",\"mentioned_list\":[\"@all\"]}}"
        else
            # 将逗号分隔的用户ID转为JSON数组
            local mentioned_json
            mentioned_json=$(echo "$WECHAT_MENTIONED_LIST" | sed 's/,/","/g')
            json_data="{\"msgtype\":\"text\",\"text\":{\"content\":\"${escaped_msg}\",\"mentioned_list\":[\"${mentioned_json}\"]}}"
        fi
    else
        json_data="{\"msgtype\":\"text\",\"text\":{\"content\":\"${escaped_msg}\"}}"
    fi

    log "WeChat sending: $json_data"
    local response
    response=$(curl -s -X POST "$WECHAT_WEBHOOK" \
        -H 'Content-Type: application/json' \
        -d "$json_data" 2>&1) || curl_rc=$?
    if [[ "$curl_rc" -ne 0 ]]; then
        log "WeChat alert failed (curl exit: $curl_rc)"
    else
        log "WeChat response: $response"
        log "WeChat alert sent"
    fi
}

# 统一发送告警（同时支持钉钉和企业微信）
send_alert() {
    local msg="$1"
    send_dingtalk "$msg"
    send_wechat "$msg"
}

# 检测无法 SSH 的节点
check_lost_hosts() {
    log "Checking lost hosts..."
    local line host
    while read -r line; do
        host=$(get_host_from_line "$line")
        (
            if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "echo ok" >/dev/null 2>&1; then
                add_error_node "$host" "ssh_unreachable"
                log "Lost host: $host"
            fi
        ) &
        wait_for_slot
    done < <(read_host_lines)
    wait
}

# 检测远端 dmesg XID 错误
check_dmesg_xid() {
    local xid_regex=""
    if [[ -n "$DMESG_XID_NUMBERS" ]]; then
        xid_regex="XID=($(echo "$DMESG_XID_NUMBERS" | sed 's/,/|/g'))"
    else
        xid_regex=$(echo "$DMESG_XID_PATTERNS" | sed 's/,/|/g')
    fi
    if [[ -z "$xid_regex" ]]; then
        log "Skipping dmesg XID check (no patterns or numbers configured)"
        return
    fi
    log "Checking dmesg XID (regex: ${xid_regex})..."
    local line host
    while read -r line; do
        host=$(get_host_from_line "$line")
        (
            xid=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "dmesg 2>/dev/null | grep -i -E '${xid_regex}' | tail -n 3" 2>/dev/null || true)
            if [[ -n "$xid" ]]; then
                add_error_node "$host" "dmesg_xid"
                log "XID reported on $host"
            fi
        ) &
        wait_for_slot
    done < <(read_host_lines)
    wait
}

# 检测远端 dmesg blocked 进程错误
check_dmesg_blocked() {
    if [[ -z "$DMESG_BLOCKED_PATTERN" ]]; then
        log "Skipping dmesg blocked check (no pattern configured)"
        return
    fi
    log "Checking dmesg blocked tasks (regex: ${DMESG_BLOCKED_PATTERN})..."
    local line host
    while read -r line; do
        host=$(get_host_from_line "$line")
        (
            blocked=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "dmesg 2>/dev/null | grep -i -E '${DMESG_BLOCKED_PATTERN}' | tail -n 3" 2>/dev/null || true)
            if [[ -n "$blocked" ]]; then
                local pid
                pid=$(echo "$blocked" | grep -oP 'task\s+[^:]+:\K[0-9]+' | head -n 1 || true)
                if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
                    local state
                    state=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "ps -o stat= -p ${pid} 2>/dev/null | tr -d ' '" 2>/dev/null || true)
                    if [[ -z "$state" ]]; then
                        log "Blocked task pid not found on $host (pid: $pid)"
                    elif [[ "$state" == *D* ]]; then
                        add_error_node "$host" "dmesg_blocked_pid_${pid}"
                        log "Blocked task reported on $host (pid: $pid, state: $state)"
                    elif [[ "$state" == *Z* ]]; then
                        add_error_node "$host" "dmesg_blocked_pid_${pid}_zombie"
                        log "Blocked task zombie on $host (pid: $pid, state: $state)"
                    else
                        log "Blocked task pid not in D/Z state on $host (pid: $pid, state: $state)"
                    fi
                else
                    if [[ -n "$pid" ]]; then
                        log "Blocked task pid invalid on $host (pid: $pid)"
                    else
                        add_error_node "$host" "dmesg_blocked"
                        log "Blocked task reported on $host"
                    fi
                fi
            fi
        ) &
        wait_for_slot
    done < <(read_host_lines)
    wait
}

# 检测远端磁盘占用
check_disk_space() {
    log "Checking disk usage (> ${DISK_THRESHOLD}%) for paths: ${DISK_PATHS}..."
    local line host usage path
    local -a disk_paths
    IFS=',' read -r -a disk_paths <<< "$DISK_PATHS"
    while read -r line; do
        host=$(get_host_from_line "$line")
        (
            for path in "${disk_paths[@]}"; do
                # trim spaces
                path="${path#${path%%[![:space:]]*}}"
                path="${path%${path##*[![:space:]]}}"
                if [[ -z "$path" ]]; then
                    continue
                fi
                usage=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "df -P '$path' 2>/dev/null | tail -1 | awk '{gsub(/%/,\"\",\$5); print \$5}'" 2>/dev/null || true)
                if [[ -n "$usage" ]] && [[ "$usage" =~ ^[0-9]+$ ]] && [[ "$usage" -gt "$DISK_THRESHOLD" ]]; then
                    local path_tag
                    path_tag=$(echo "$path" | sed -E 's#[^a-zA-Z0-9]+#_#g')
                    add_error_node "$host" "disk_usage_${path_tag}_${usage}%"
                    log "Disk high on $host ($path): ${usage}%"
                fi
            done
        ) &
        wait_for_slot
    done < <(read_host_lines)
    wait
}

# 检测远端 musaInfo 执行状态
check_musainfo() {
    log "Checking musaInfo on all nodes..."
    # 创建 musaInfo 日志目录（带时间戳）
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_dir="${MUSAINFO_LOG_DIR}/${timestamp}"
    mkdir -p "$log_dir"
    log "musaInfo logs will be saved to: $log_dir"

    local line host
    local error_count=0
    while read -r line; do
        host=$(get_host_from_line "$line")
        (
            local host_log="${log_dir}/${host}.log"
            local exit_code=0
            # 执行 musaInfo 并保存输出
            ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "musaInfo" > "$host_log" 2>&1 || exit_code=$?

            if [[ "$exit_code" -ne 0 ]]; then
                add_error_node "$host" "musainfo_failed"
                log "musaInfo failed on $host (exit code: $exit_code)"
                echo "EXIT_CODE: $exit_code" >> "$host_log"
            fi
        ) &
        wait_for_slot
    done < <(read_host_lines)
    wait

    # 统计失败节点数
    error_count=$(grep -c "musainfo_failed" "$ERROR_FILE" 2>/dev/null || true)
    error_count=${error_count:-0}
    if [[ "$error_count" -gt 0 ]]; then
        log "musaInfo check completed: $error_count nodes failed"
    else
        log "musaInfo check completed: all nodes passed"
    fi
}

# 运行 mccl 网络检查并标记异常 IP
# 可通过 NETCHECK_TIMEOUT 环境变量或配置项设置超时（默认 300 秒）
NETCHECK_TIMEOUT="${NETCHECK_TIMEOUT:-300}"
SKIP_INITIAL_NETCHECK="${SKIP_INITIAL_NETCHECK:-0}"

run_netcheck() {
    if [[ ! -f "$MCCL_BENCH" ]]; then
        log "mccl_bench.sh not found: $MCCL_BENCH, skipping netcheck"
        return
    fi

    # 在运行 mccl_bench 前先停止所有 GPU 进程，确保资源可用
    log "Stopping all GPU processes before netcheck..."
    if [[ -f "$STOP_ALL" ]]; then
        timeout 120 bash "$STOP_ALL" "$HOSTFILE" "$MAX_PARALLEL" || true
        sleep 30  # 等待进程完全退出
    else
        log "Warning: stop_all.sh not found, proceeding without cleanup"
    fi

    log "Running mccl netcheck (timeout: ${NETCHECK_TIMEOUT}s)..."
    local tmp_out
    tmp_out=$(mktemp)

    # 使用 timeout 命令防止无限等待，并显示输出以便用户看到进度
    local exit_code=0
    timeout "$NETCHECK_TIMEOUT" bash "$MCCL_BENCH" "$HOSTFILE" --netcheck 2>&1 | tee "$tmp_out" || exit_code=$?

    local timed_out=0
    if [[ "$exit_code" -eq 124 ]]; then
        timed_out=1
        log "Netcheck timed out after ${NETCHECK_TIMEOUT}s"
    elif [[ "$exit_code" -ne 0 ]]; then
        log "Netcheck exited with code $exit_code"
    fi

    local bad_ips
    # 使用 || true 防止 grep 无匹配时返回非零导致脚本退出
    bad_ips=$(grep -Eo 'Phase 2 FAIL.*([0-9]{1,3}\.){3}[0-9]{1,3}' "$tmp_out" 2>/dev/null | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' | sort -u || true)
    if [[ -z "$bad_ips" ]]; then
        bad_ips=$(awk '/Problematic IPs/{flag=1; next} flag && $1 ~ /^-$/ {next} flag && $1 ~ /^-/ {print $2} flag && $1 ~ /^[0-9]/{print $1}' "$tmp_out" 2>/dev/null | sort -u || true)
    fi
    local need_dump=0
    if [[ "$timed_out" -eq 1 || "$exit_code" -ne 0 || -n "$bad_ips" ]]; then
        need_dump=1
    fi
    # 如果超时且存在未完成的 group，则判定为故障
    if [[ "$timed_out" -eq 1 ]]; then
        local netcheck_log_dir
        netcheck_log_dir=$(grep -m1 -oE 'Log directory: .*' "$tmp_out" | sed 's/Log directory: //' 2>/dev/null || true)
        if [[ -n "$netcheck_log_dir" && "$netcheck_log_dir" != /* ]]; then
            netcheck_log_dir="${SCRIPT_DIR}/${netcheck_log_dir#./}"
        fi

        mapfile -t host_lines < <(read_host_lines)
        local num_hosts
        num_hosts=${#host_lines[@]}
        local pair_count=$((num_hosts / 2))
        if [[ "$pair_count" -gt 0 ]]; then
            local completed_groups
            completed_groups=$(grep -oE '\[Group p1_[0-9]+' "$tmp_out" 2>/dev/null | sed 's/\[Group p1_//' | sort -u || true)
            for ((gid=1; gid<=pair_count; gid++)); do
                if ! echo "$completed_groups" | grep -qx "$gid"; then
                    local host1 host2 ip1 ip2
                    host1="${host_lines[$(( (gid-1)*2 ))]}"
                    host2="${host_lines[$(( (gid-1)*2 + 1 ))]}"
                    ip1=$(get_host_from_line "$host1")
                    ip2=$(get_host_from_line "$host2")
                    if [[ -n "$ip1" ]]; then
                        add_error_node "$ip1" "mccl_netcheck_timeout_group_${gid}"
                    fi
                    if [[ -n "$ip2" ]]; then
                        add_error_node "$ip2" "mccl_netcheck_timeout_group_${gid}"
                    fi
                    log "Netcheck timeout: group p1_${gid} no result, marking ${ip1} ${ip2}"
                fi
            done
        fi
    fi

    if [[ "$need_dump" -eq 1 ]]; then
        log "Netcheck detail output (last 200 lines):"
        tail -n 200 "$tmp_out" 2>/dev/null | while IFS= read -r line; do
            log "netcheck: $line"
        done
    fi

    rm -f "$tmp_out"

    if [[ -n "$bad_ips" ]]; then
        for ip in $bad_ips; do
            add_error_node "$ip" "mccl_netcheck"
        done
        log "Netcheck bad IPs: $bad_ips"
    else
        log "Netcheck completed, no bad IPs found"
    fi
}

# 收集所有节点的 dmesg -T 信息并打包到日志目录
collect_dmesg_logs() {
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local dmesg_dir="${LOG_DIR}/dmesg_${timestamp}"
    mkdir -p "$dmesg_dir"
    log "Collecting dmesg -T from all nodes to: $dmesg_dir"

    local hosts_file="$RUN_HOSTFILE"
    if [[ ! -f "$hosts_file" ]]; then
        hosts_file="$HOSTFILE"
    fi

    # 统计期望收集的节点数（去重、去空行）
    local expected_hosts
    expected_hosts=$(mktemp)
    local expected_count=0
    while read -r line; do
        local host
        host=$(get_host_from_line "$line")
        # 去除 CRLF 和前后空白
        host=$(echo "$host" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$host" ]] && continue
        echo "$host" >> "$expected_hosts"
        ((expected_count++)) || true
    done < <(grep -v '^#' "$hosts_file" | awk 'NF {print}')

    # 去重
    sort -u -o "$expected_hosts" "$expected_hosts"
    expected_count=$(wc -l < "$expected_hosts" | tr -d ' ')

    log "Expected to collect dmesg from $expected_count hosts"

    # 并行收集 dmesg
    local pids=()
    while read -r host; do
        [[ -z "$host" ]] && continue
        (
            local host_log="${dmesg_dir}/${host}_${timestamp}.dmesg.log"
            local ssh_output=""
            local ssh_rc=0
            ssh_output=$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=5 "$host" "dmesg -T 2>&1" 2>&1) || ssh_rc=$?
            if [[ $ssh_rc -eq 0 && -n "$ssh_output" ]]; then
                echo "$ssh_output" > "$host_log"
            else
                # SSH 失败或输出为空，创建失败标记文件
                echo "FAILED: ssh exit code=$ssh_rc, output: $ssh_output" > "${host_log}.failed"
            fi
        ) &
        pids+=($!)
        # 控制并发数
        if [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; then
            # 等待任一进程完成
            wait -n 2>/dev/null || wait "${pids[0]}" 2>/dev/null || true
            # 清理已完成的 PID
            local new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            pids=("${new_pids[@]}")
        fi
    done < "$expected_hosts"
    # 等待所有剩余进程完成
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    # 统计实际收集到的文件数（排除 .failed 文件）
    local collected_count
    collected_count=$(find "$dmesg_dir" -maxdepth 1 -name '*.dmesg.log' -type f 2>/dev/null | wc -l)
    collected_count=${collected_count:-0}

    log "Collected dmesg from $collected_count/$expected_count hosts"

    # 如果有缺失，生成缺失主机列表
    if [[ "$collected_count" -lt "$expected_count" ]]; then
        local missing_report="${dmesg_dir}/missing_hosts.txt"
        while read -r host; do
            [[ -z "$host" ]] && continue
            local host_log="${dmesg_dir}/${host}_${timestamp}.dmesg.log"
            # 检查是否成功收集（文件存在且没有对应的 .failed 文件）
            if [[ ! -f "$host_log" ]] || [[ -f "${host_log}.failed" ]]; then
                echo "$host" >> "$missing_report"
            fi
        done < "$expected_hosts"
        if [[ -f "$missing_report" ]]; then
            local missing_count
            missing_count=$(wc -l < "$missing_report" | tr -d ' ')
            log "Warning: $missing_count hosts missing dmesg logs, see: $missing_report"
        fi
    fi

    rm -f "$expected_hosts"

    # 打包 dmesg 日志
    local tar_file="${LOG_DIR}/dmesg_${timestamp}.tar.gz"
    if command -v tar >/dev/null 2>&1; then
        tar -czf "$tar_file" -C "$LOG_DIR" "dmesg_${timestamp}" 2>/dev/null || true
        if [[ -f "$tar_file" ]]; then
            log "dmesg logs archived to: $tar_file"
            # 可选：删除原始目录以节省空间
            # rm -rf "$dmesg_dir"
        fi
    else
        log "tar not found, dmesg logs saved to: $dmesg_dir"
    fi
}

# 故障检测总入口
# 参数: $1 - 如果为 "initial" 则检查 SKIP_INITIAL_NETCHECK
fault_detect() {
    local mode="${1:-}"
    log "=== Fault detection start ==="
    check_lost_hosts
    check_dmesg_xid
    check_dmesg_blocked
    check_disk_space
    check_musainfo
    if [[ "$mode" == "initial" && "$SKIP_INITIAL_NETCHECK" -eq 1 ]]; then
        log "Skipping initial mccl netcheck (--skip-initial-netcheck)"
    else
        run_netcheck
    fi
    # scan_log_errors_on_fault 已在告警前单独调用，这里不重复
    # 收集所有节点的 dmesg 信息
    collect_dmesg_logs
    log "=== Fault detection end ==="
}

start_hang_detect() {
    if [[ "$TRAIN_START_TIME" -le 0 ]]; then
        log "Skip hang detector start because training has not started"
        return 0
    fi

    if [[ ! -f "$HANG_DETECT_SCRIPT" ]]; then
        log "Warning: hang_detect.sh not found, skip hang detection: $HANG_DETECT_SCRIPT"
        HANG_DETECT_PID=0
        return 0
    fi

    if [[ "$HANG_DETECT_PID" -gt 0 ]] && kill -0 "$HANG_DETECT_PID" 2>/dev/null; then
        kill "$HANG_DETECT_PID" 2>/dev/null || true
        wait "$HANG_DETECT_PID" 2>/dev/null || true
    fi

    log "Starting hang detector: $HANG_DETECT_SCRIPT --hostfile $HOSTFILE --run-hostfile $RUN_HOSTFILE --logdir $LOG_DIR --poll-interval $POLL_INTERVAL --hang-minutes $HANG_MINUTES"
    bash "$HANG_DETECT_SCRIPT" \
        --hostfile "$HOSTFILE" \
        --run-hostfile "$RUN_HOSTFILE" \
        --logdir "$LOG_DIR" \
        --poll-interval "$POLL_INTERVAL" \
        --hang-minutes "$HANG_MINUTES" \
        --alert-on-hang \
        >> "$RUN_LOG" 2>&1 &
    HANG_DETECT_PID=$!
    log "hang_detect PID: $HANG_DETECT_PID"
}

stop_hang_detect() {
    if [[ "${HANG_DETECT_PID:-0}" -gt 0 ]] && kill -0 "$HANG_DETECT_PID" 2>/dev/null; then
        log "Stopping hang_detect process (PID: $HANG_DETECT_PID)..."
        kill "$HANG_DETECT_PID" 2>/dev/null || true
        wait "$HANG_DETECT_PID" 2>/dev/null || true
    fi
    HANG_DETECT_PID=0
}

# 启动训练（dist_run.sh）
start_train() {
    if ! build_run_hostfile; then
        return 1
    fi
    if [[ "$START_TRAIN" -eq 1 ]]; then
        if [[ ! -f "$DIST_RUN" ]]; then
            log "Error: dist_run.sh not found: $DIST_RUN"
            return 1
        fi
        # 重启训练时再次确认 128 默认（避免中途 env 被清）
        if [[ "$WORLDSIZE" -ge 128 ]]; then
            export LLM_PRETRAIN_DIST_TRAIN="${LLM_PRETRAIN_DIST_TRAIN:-scripts/dist_train_megatron_ws128.sh}"
            export MUSA_PRETRAIN_ENTRY="${MUSA_PRETRAIN_ENTRY:-musa_pretrain_ws128.sh}"
        fi
        local dist_run_name
        dist_run_name=$(basename "$DIST_RUN")
        local extra_args=()
        if [[ -n "$LOG_DIR" ]]; then
            extra_args+=("--logdir" "$LOG_DIR")
        fi
        if [[ -n "$OUTPUT_DIR" ]]; then
            extra_args+=("--output-dir" "$OUTPUT_DIR")
        fi
        if [[ -n "$WORKLOAD" ]]; then
            log "Starting dist_run: $DIST_RUN $RUN_HOSTFILE $WORKLOAD ${extra_args[*]} (ENTRY=${MUSA_PRETRAIN_ENTRY:-} DIST_TRAIN=${LLM_PRETRAIN_DIST_TRAIN:-})"
            bash "$DIST_RUN" "$RUN_HOSTFILE" "$WORKLOAD" "${extra_args[@]}" &
            RUN_PID=$!
        else
            log "Starting dist_run: $DIST_RUN $RUN_HOSTFILE ${extra_args[*]} (ENTRY=${MUSA_PRETRAIN_ENTRY:-} DIST_TRAIN=${LLM_PRETRAIN_DIST_TRAIN:-})"
            bash "$DIST_RUN" "$RUN_HOSTFILE" "${extra_args[@]}" &
            RUN_PID=$!
        fi
        log "dist_run PID: $RUN_PID"
        TRAIN_START_TIME=$(date +%s)
        reset_grace_ready_hosts
        log "Train start time recorded, grace period: ${STARTUP_GRACE}s"
        start_hang_detect
    fi
}

# 通过进程数判断恢复是否成功
# 返回 0 表示恢复成功（所有进程存在），返回 1 表示恢复失败
check_recovery_by_process() {
    local max_wait="${1:-$STARTUP_GRACE}"
    local interval=10
    local waited=0
    # 等待训练进程拉起
    while [[ "$waited" -lt "$max_wait" ]]; do
        sleep "$interval"
        waited=$((waited + interval))
        if ! check_process_exit; then
            return 0
        fi
    done
    # 超时后再做一次确认
    if ! check_process_exit; then
        return 0
    fi
    return 1
}

# 扫描日志错误关键字
# 结果保存到全局变量 LAST_MATCHED_ERROR
LAST_MATCHED_ERROR=""
LAST_MATCHED_FILE=""
check_log_errors() {
    LAST_MATCHED_ERROR=""
    LAST_MATCHED_FILE=""
    if [[ ! -d "$LOG_DIR" ]]; then
        return 1
    fi
    local regex
    regex=$(echo "$LOG_ERROR_PATTERNS" | sed 's/,/|/g')
    local nan_pattern
    nan_pattern="${LOG_NAN_PATTERN:-}"
    if [[ -z "$regex" && -z "$nan_pattern" ]]; then
        return 1
    fi
    local matched=1
    local window_min
    window_min=$((POLL_INTERVAL / 60 + 1))
    local files
    # 仅分析 LOG_DIR 目录下的文本文件（不递归子目录）
    files=$(find "$LOG_DIR" -maxdepth 1 -type f -mmin "-${window_min}" 2>/dev/null || true)
    if [[ -n "$files" ]]; then
        # 按优先级逐个匹配 pattern（排列顺序决定优先级，Traceback 应放最后）
        if [[ -n "$regex" ]]; then
            IFS=',' read -r -a _patterns <<< "$LOG_ERROR_PATTERNS"
            for _pat in "${_patterns[@]}"; do
                _pat="${_pat#${_pat%%[![:space:]]*}}"
                _pat="${_pat%${_pat##*[![:space:]]}}"
                [[ -z "$_pat" ]] && continue
                local grep_result
                grep_result=$(echo "$files" | xargs -r grep -I -i -l -E "$_pat" 2>/dev/null | head -n 1 || true)
                if [[ -n "$grep_result" ]]; then
                    LAST_MATCHED_FILE=$(basename "$grep_result")
                    LAST_MATCHED_ERROR=$(grep -I -i -E "$_pat" "$grep_result" 2>/dev/null | head -n 1 | head -c 200 || true)
                    log "check_log_errors: found pattern '$_pat' in: $grep_result"
                    log "check_log_errors: matched content: $LAST_MATCHED_ERROR"
                    matched=0
                    break
                fi
            done
        fi
        if [[ "$matched" -ne 0 && -n "$nan_pattern" ]]; then
            local grep_result
            grep_result=$(echo "$files" | xargs -r grep -I -l -F "$nan_pattern" 2>/dev/null | head -n 1 || true)
            if [[ -n "$grep_result" ]]; then
                LAST_MATCHED_FILE=$(basename "$grep_result")
                LAST_MATCHED_ERROR=$(grep -I -F "$nan_pattern" "$grep_result" 2>/dev/null | head -n 1 | head -c 200 || true)
                log "check_log_errors: found nan pattern in: $grep_result"
                log "check_log_errors: matched content: $LAST_MATCHED_ERROR"
                matched=0
            fi
        fi
    else
        log "check_log_errors: no files modified in last ${window_min} min"
    fi
    # 若窗口内未命中 NaN，回退扫描最近文件（避免日志更新时间不在窗口内导致漏检）
    if [[ "$matched" -ne 0 && -n "$nan_pattern" ]]; then
        local recent_files
        recent_files=$(find "$LOG_DIR" -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 200 | awk '{print $2}' || true)
        if [[ -n "$recent_files" ]]; then
            local grep_result
            grep_result=$(echo "$recent_files" | xargs -r grep -I -l -F "$nan_pattern" 2>/dev/null | head -n 1 || true)
            if [[ -n "$grep_result" ]]; then
                LAST_MATCHED_FILE=$(basename "$grep_result")
                LAST_MATCHED_ERROR=$(grep -I -F "$nan_pattern" "$grep_result" 2>/dev/null | head -n 1 | head -c 200 || true)
                log "check_log_errors: found nan pattern (fallback) in: $grep_result"
                log "check_log_errors: matched content: $LAST_MATCHED_ERROR"
                matched=0
            fi
        fi
    fi
    return $matched
}

# 检测训练是否正常结束（日志包含关键词）
check_training_finished() {
    if [[ ! -d "$LOG_DIR" ]]; then
        return 1
    fi
    local pattern
    pattern="$TRAIN_FINISHED_PATTERN"
    if [[ -z "$pattern" ]]; then
        return 1
    fi
    local files
    files=$(find "$LOG_DIR" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 200 | awk '{print $2}' || true)
    if [[ -z "$files" ]]; then
        return 1
    fi
    while read -r file; do
        [[ -z "$file" ]] && continue
        if grep -i -m1 -E "$pattern" "$file" >/dev/null 2>&1; then
            return 0
        fi
    done <<< "$files"
    return 1
}

# 故障后扫描训练日志错误关键字（输出命中行）
# 结果保存到全局变量 LAST_ERROR_SUMMARY
LAST_ERROR_SUMMARY=""
scan_log_errors_on_fault() {
    LAST_ERROR_SUMMARY=""
    if [[ ! -d "$LOG_DIR" ]]; then
        return
    fi
    local regex
    regex=$(echo "$LOG_ERROR_PATTERNS" | sed 's/,/|/g')
    local nan_pattern
    nan_pattern="${LOG_NAN_PATTERN:-}"
    if [[ -z "$regex" && -z "$nan_pattern" ]]; then
        return
    fi
    log "Scanning training logs for errors after fault (patterns: ${LOG_ERROR_PATTERNS}, nan: ${nan_pattern})..."

    # 使用与 check_log_errors 相同的文件搜索范围
    local window_min
    window_min=$((POLL_INTERVAL / 60 + 1))
    local files
    files=$(find "$LOG_DIR" -type f -mmin "-${window_min}" 2>/dev/null || true)
    # 若窗口内无文件，回退到最近修改的文件
    if [[ -z "$files" ]]; then
        files=$(find "$LOG_DIR" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 50 | awk '{print $2}' || true)
    fi
    if [[ -z "$files" ]]; then
        log "No log files found in $LOG_DIR"
        return
    fi
    local matched=0
    local summary_lines=0
    local max_summary_lines=20
    while read -r file; do
        if [[ -z "$file" ]]; then
            continue
        fi
        if [[ -n "$regex" ]]; then
            # 使用 grep -B3 -A3 显示前后3行上下文
            local error_output
            error_output=$(grep -i -E "$regex" -n -B3 -A3 "$file" 2>/dev/null | head -n 50 || true)
            if [[ -n "$error_output" ]]; then
                local filename
                filename=$(basename "$file")
                log "log_error in file: ${file}"
                log "----------------------------------------"
                echo "$error_output" | while IFS= read -r line; do
                    log "  $line"
                done
                log "----------------------------------------"
                matched=1
                # 添加到摘要（限制行数）
                if [[ $summary_lines -lt $max_summary_lines ]]; then
                    LAST_ERROR_SUMMARY="${LAST_ERROR_SUMMARY}[${filename}]\n"
                    local first_error
                    first_error=$(echo "$error_output" | grep -E "$regex" | head -n 2)
                    LAST_ERROR_SUMMARY="${LAST_ERROR_SUMMARY}${first_error}\n"
                    summary_lines=$((summary_lines + 3))
                fi
            fi
        fi
        if [[ -n "$nan_pattern" ]]; then
            local nan_output
            nan_output=$(grep -F "$nan_pattern" -n -B3 -A3 "$file" 2>/dev/null | head -n 50 || true)
            if [[ -n "$nan_output" ]]; then
                local filename
                filename=$(basename "$file")
                log "log_nan in file: ${file}"
                log "----------------------------------------"
                echo "$nan_output" | while IFS= read -r line; do
                    log "  $line"
                done
                log "----------------------------------------"
                matched=1
                # 添加到摘要（限制行数）
                if [[ $summary_lines -lt $max_summary_lines ]]; then
                    LAST_ERROR_SUMMARY="${LAST_ERROR_SUMMARY}[${filename}] NaN detected\n"
                    summary_lines=$((summary_lines + 1))
                fi
            fi
        fi
    done <<< "$files"
    if [[ "$matched" -eq 0 ]]; then
        log "No error patterns found in recent logs"
    fi
}

# 检测日志是否长时间无更新
check_log_stale() {
    local latest
    latest=$(get_latest_mtime)
    if [[ "$latest" -eq 0 ]]; then
        return 1
    fi
    local now
    now=$(date +%s)
    local diff
    diff=$((now - latest))
    if [[ "$diff" -gt $((STALE_MIN * 60)) ]]; then
        return 0
    fi
    return 1
}

# 获取日志目录最新文件的修改时间
get_latest_mtime() {
    if [[ ! -d "$LOG_DIR" ]]; then
        echo 0
        return
    fi
    local latest
    latest=$(find "$LOG_DIR" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 || true)
    if [[ -z "$latest" ]]; then
        echo 0
    else
        printf "%.0f" "$latest"
    fi
}

# 监控主循环：检测异常 -> 告警 -> 处置
monitor_loop() {
    log "Monitoring log dir: $LOG_DIR (stale>${STALE_MIN}m, poll=${POLL_INTERVAL}s, grace=${STARTUP_GRACE}s, hang>${HANG_MINUTES}m)"
    local last_alert=0
    while true; do
        local reasons=()
        if [[ "${HANG_DETECT_PID:-0}" -gt 0 ]] && ! kill -0 "$HANG_DETECT_PID" 2>/dev/null; then
            local hang_rc=0
            wait "$HANG_DETECT_PID" 2>/dev/null || hang_rc=$?
            HANG_DETECT_PID=0
            if [[ "$hang_rc" -eq 100 ]]; then
                log "Hang detector reported training hang"
                reasons+=("training_hang")
            else
                log "Hang detector exited unexpectedly with rc=$hang_rc, restarting"
                start_hang_detect
            fi
        fi
        reload_config_if_needed
        heartbeat_cluster_lock
        local now
        now=$(date +%s)

        # 检查是否过了启动宽限期
        local grace_elapsed=0
        if [[ "$TRAIN_START_TIME" -gt 0 ]]; then
            grace_elapsed=$((now - TRAIN_START_TIME))
        fi

        # 只有在宽限期后才检测进程状态
        if [[ "$grace_elapsed" -ge "$STARTUP_GRACE" ]]; then
            if check_process_exit; then
                if check_training_finished; then
                    log "Process exit detected but training finished found in logs, treat as completed"
                    send_alert "【训练完成】
时间: $(date '+%F %T')
状态: 训练正常完成
日志目录: $LOG_DIR"
                    exit 0
                fi
                reasons+=("process_exit")
            fi
        else
            log "Startup grace (${grace_elapsed}s/${STARTUP_GRACE}s), checking ready hosts"
            local ready_hosts
            ready_hosts=$(get_process_ready_hosts)
            local current_ready_count=0
            if [[ -n "$ready_hosts" ]]; then
                # 按行计数（每行一个 IP），避免空格分词膨胀
                current_ready_count=$(echo "$ready_hosts" | wc -l | tr -d ' ')
            fi
            local ready_count=$current_ready_count
            local ready_target="${WORLDSIZE:-0}"

            # 启动宽限期内，每次全量检测所有节点是否就绪
            if [[ "$ready_target" -gt 0 && "$ready_count" -ge "$ready_target" ]]; then
                TRAIN_START_TIME=$((now - STARTUP_GRACE))
                log "All nodes ready during grace period, exiting grace early"
            fi
        fi

        if check_log_stale; then
            log "Log stale: FAULT - no update > ${STALE_MIN}m"
            reasons+=("log_stale")
        fi
        local nan_detected=0
        local nan_sample=""
        if check_log_errors; then
            log "Log errors: FAULT - matched pattern"
            reasons+=("log_error")
            # 检查是否是 nan 导致的错误
            if [[ -n "${LOG_NAN_PATTERN:-}" ]]; then
                local window_min
                window_min=$((POLL_INTERVAL / 60 + 1))
                local files
                files=$(find "$LOG_DIR" -type f -mmin "-${window_min}" 2>/dev/null || true)
                if [[ -n "$files" ]]; then
                    nan_sample=$(echo "$files" | xargs -r grep -F "$LOG_NAN_PATTERN" -m 1 2>/dev/null | head -n 1 || true)
                    if [[ -n "$nan_sample" ]]; then
                        nan_detected=1
                        reasons+=("loss_nan")
                    fi
                fi
            fi
        fi

        if [[ ${#reasons[@]} -eq 0 ]]; then
            log "Monitor: no err"
        fi

        if [[ ${#reasons[@]} -gt 0 ]]; then
            local reason_str
            reason_str=$(IFS=','; echo "${reasons[*]}")
            local alert_time
            alert_time=$(date '+%F %T')

            # 先扫描日志错误，获取错误摘要
            scan_log_errors_on_fault

            # 构建告警信息（包含具体匹配内容）
            local log_dir_name
            log_dir_name=$(basename "$LOG_DIR")
            local action_desc="正在自动恢复..."
            if [[ "$HANG_NO_KILL" -eq 1 && "$reason_str" == "training_hang" ]]; then
                action_desc="已按配置保留现场，不杀进程"
            fi
            local alert_msg="[故障告警] ${alert_time} 原因:${reason_str} 重启#$((RESTART_COUNT + 1)) 日志:${log_dir_name}"

            # 添加匹配的文件和错误内容
            if [[ -n "$LAST_MATCHED_FILE" ]]; then
                alert_msg="${alert_msg} 文件:${LAST_MATCHED_FILE}"
            fi
            if [[ -n "$LAST_MATCHED_ERROR" ]]; then
                # 截取前150字符避免消息过长
                local error_short
                error_short=$(echo "$LAST_MATCHED_ERROR" | head -c 150)
                alert_msg="${alert_msg} 错误:${error_short}"
            fi
            alert_msg="${alert_msg} ${action_desc}"

            if [[ $((now - last_alert)) -gt $((STALE_MIN * 60)) ]]; then
                send_alert "$alert_msg"
                last_alert=$now
            fi
            if [[ "$HANG_NO_KILL" -eq 1 && "$reason_str" == "training_hang" ]]; then
                log "Hang detected with --hang-no-kill, keep training processes alive after dump"
                stop_hang_detect
            else
                stop_hang_detect
                fault_detect
                if [[ "$START_TRAIN" -eq 1 ]]; then
                    ((RESTART_COUNT++)) || true
                    LOG_DIR="${LOG_DIR_BASE}_r${RESTART_COUNT}"
                    mkdir -p "$LOG_DIR"
                    log "Restart #${RESTART_COUNT}, updated log dir: $LOG_DIR"
                    stop_train || true
                    if start_train; then
                        if check_recovery_by_process "$STARTUP_GRACE"; then
                            send_alert "[恢复成功] $(date '+%F %T') 重启#${RESTART_COUNT} 原因:${reason_str}"
                        else
                            send_alert "[恢复失败] $(date '+%F %T') 重启#${RESTART_COUNT} 原因:${reason_str} 请人工检查"
                        fi
                    else
                        send_alert "[启动失败] $(date '+%F %T') 重启#${RESTART_COUNT} 原因:${reason_str} 请人工检查"
                    fi
                else
                    build_run_hostfile || true
                fi
            fi
        fi
        if [[ "${HANG_DETECT_PID:-0}" -eq 0 ]]; then
            start_hang_detect
        fi
        sleep "$POLL_INTERVAL" || true
    done
}

# 启动前先做一次故障检测（标记为 initial 模式，可跳过 netcheck）
if [[ "$SKIP_INITIAL_FAULT_DETECT" -ne 1 ]]; then
    fault_detect "initial"
fi
# 启动训练或仅生成 hostfile
if [[ "$START_TRAIN" -eq 1 && "$SKIP_INITIAL_START" -eq 0 ]]; then
    start_train
else
    build_run_hostfile
fi
# 进入持续监控
monitor_loop

