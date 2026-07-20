#!/bin/bash

# 停止所有节点上占用 GPU 的进程
# Usage: ./stop_all.sh HOSTFILE [MAX_PARALLEL]
# 通过 mthreads-gmi 获取 GPU 上的 PID 并 kill

HOSTFILE=$1
MAX_PARALLEL=${2:-32}

if [[ -z "$HOSTFILE" || ! -f "$HOSTFILE" ]]; then
    echo "Usage: $0 HOSTFILE [MAX_PARALLEL]"
    exit 1
fi

NUM_NODES=$(grep -v '^#\|^$' "$HOSTFILE" | wc -l)
echo "Stopping GPU processes on $NUM_NODES nodes (parallel: $MAX_PARALLEL)..."
echo ""

# 临时文件用于收集结果
RESULT_FILE=$(mktemp)
trap "rm -f $RESULT_FILE" EXIT

wait_for_slot() {
    local max="${1:-$MAX_PARALLEL}"
    while true; do
        local running
        running=$(jobs -pr | wc -l | tr -d ' ')
        if [[ "$running" -lt "$max" ]]; then
            break
        fi
        sleep 0.2
    done
}

hostlist=$(grep -v '^#\|^$' "$HOSTFILE" | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
    (
        # 远程执行：通过 mthreads-gmi 获取 GPU PID 并 kill
        result=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s 2>/dev/null <<'EOF'
            pids=$(mthreads-gmi 2>/dev/null | awk '
                /^Processes:/ { in_proc=1; next }
                in_proc && /^[0-9]+[[:space:]]+[0-9]+/ { print $2 }
            ' | sort -u)
            
            if [[ -z "$pids" ]]; then
                echo "NONE"
                exit 0
            fi
            
            killed=""
            failed=""
            for pid in $pids; do
                if kill -0 "$pid" 2>/dev/null; then
                    if kill -9 "$pid" 2>/dev/null; then
                        killed="$killed $pid"
                    else
                        failed="$failed $pid"
                    fi
                fi
            done
            
            # 输出格式: KILLED:pid1,pid2|FAILED:pid3,pid4
            killed=$(echo $killed | tr ' ' ',' | sed 's/^,//')
            failed=$(echo $failed | tr ' ' ',' | sed 's/^,//')
            echo "KILLED:${killed}|FAILED:${failed}"
EOF
        )
        
        if [[ -z "$result" ]]; then
            printf "%-20s SSH_FAILED\n" "$host" >> "$RESULT_FILE"
        elif [[ "$result" == "NONE" ]]; then
            printf "%-20s no_gpu_process\n" "$host" >> "$RESULT_FILE"
        else
            killed=$(echo "$result" | grep -oP 'KILLED:\K[^|]*' || true)
            failed=$(echo "$result" | grep -oP 'FAILED:\K.*' || true)
            if [[ -n "$killed" ]]; then
                printf "%-20s killed: %s\n" "$host" "$killed" >> "$RESULT_FILE"
            fi
            if [[ -n "$failed" ]]; then
                printf "%-20s FAILED: %s\n" "$host" "$failed" >> "$RESULT_FILE"
            fi
            if [[ -z "$killed" && -z "$failed" ]]; then
                printf "%-20s no_gpu_process\n" "$host" >> "$RESULT_FILE"
            fi
        fi
    ) &
    wait_for_slot
done
wait

# 输出排序后的结果
echo "=============================================="
printf "%-20s %s\n" "HOST" "STATUS"
echo "----------------------------------------------"

# 先输出有 killed 的
grep -E "killed:" "$RESULT_FILE" 2>/dev/null | sort || true
# 再输出 no_gpu_process 的
grep -E "no_gpu_process" "$RESULT_FILE" 2>/dev/null | sort || true
# 最后输出失败的
grep -E "FAILED:|SSH_FAILED" "$RESULT_FILE" 2>/dev/null | sort || true

echo "=============================================="

# 统计
total=$(wc -l < "$RESULT_FILE" | tr -d ' \n')
killed_count=$(grep -c "killed:" "$RESULT_FILE" 2>/dev/null | tr -d ' \n' || echo 0)
none_count=$(grep -c "no_gpu_process" "$RESULT_FILE" 2>/dev/null | tr -d ' \n' || echo 0)
failed_count=$(grep -cE "FAILED:|SSH_FAILED" "$RESULT_FILE" 2>/dev/null | tr -d ' \n' || echo 0)

echo "Summary: $total nodes processed"
echo "  - Killed GPU processes: $killed_count nodes"
echo "  - No GPU process found: $none_count nodes"
if [[ "$failed_count" -gt 0 ]]; then
    echo "  - Failed: $failed_count nodes"
fi
