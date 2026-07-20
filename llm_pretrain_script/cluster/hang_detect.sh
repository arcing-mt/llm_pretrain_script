#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

HOSTFILE=""
LOG_DIR=""
RUN_HOSTFILE=""
POLL_INTERVAL=60
HANG_MINUTES=10
MASTER_RANK=0
DUMP_BASE_DIR="/home/jd/blake"
ALERT_ON_HANG=0
HANG_REASON=""

usage() {
    cat <<EOF
Usage: $0 --hostfile HOSTFILE --logdir LOG_DIR [options]

Options:
  --hostfile PATH           Hostfile path
  --logdir PATH             Training log directory
  --run-hostfile PATH       Runtime hostfile path (default: hostfile)
  --poll-interval SEC       Poll interval in seconds (default: 60)
  --hang-minutes MIN        Hang threshold in minutes since last step/loss log (default: 10)
  --master-rank N           Master rank index in hostfile (default: 0)
  --dump-base-dir PATH      Remote dump base dir (default: /home/jd/blake)
  --alert-on-hang           Print HANG_DETECTED marker to stdout when hang detected
  -h, --help                Show help
EOF
}

log() {
    echo "[$(date '+%F %T')] [hang_detect] $*" >&2
}

resolve_path() {
    local path="$1"
    if [[ -z "$path" ]]; then
        echo ""
        return
    fi
    if [[ "$path" != /* ]]; then
        path="${SCRIPT_DIR}/${path#./}"
    fi
    echo "$path"
}

get_host_from_line() {
    echo "$1" | awk '{print $1}'
}

read_host_lines() {
    local hosts_file="$1"
    grep -v '^#' "$hosts_file" | awk 'NF {print}'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hostfile) HOSTFILE="$2"; shift 2 ;;
        --logdir) LOG_DIR="$2"; shift 2 ;;
        --run-hostfile) RUN_HOSTFILE="$2"; shift 2 ;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --hang-minutes) HANG_MINUTES="$2"; shift 2 ;;
        --master-rank) MASTER_RANK="$2"; shift 2 ;;
        --dump-base-dir) DUMP_BASE_DIR="$2"; shift 2 ;;
        --alert-on-hang) ALERT_ON_HANG=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$HOSTFILE" || -z "$LOG_DIR" ]]; then
    usage
    exit 1
fi

HOSTFILE=$(resolve_path "$HOSTFILE")
LOG_DIR=$(resolve_path "$LOG_DIR")
if [[ -z "$RUN_HOSTFILE" ]]; then
    RUN_HOSTFILE="$HOSTFILE"
else
    RUN_HOSTFILE=$(resolve_path "$RUN_HOSTFILE")
fi

if [[ ! -f "$HOSTFILE" ]]; then
    log "hostfile not found: $HOSTFILE"
    exit 1
fi

if [[ ! -f "$RUN_HOSTFILE" ]]; then
    log "run hostfile not found, fallback to hostfile: $RUN_HOSTFILE"
    RUN_HOSTFILE="$HOSTFILE"
fi

if [[ ! -d "$LOG_DIR" ]]; then
    log "log dir not found: $LOG_DIR"
    exit 1
fi

find_master_host() {
    local idx=0
    local line
    while read -r line; do
        [[ -z "$line" ]] && continue
        if [[ "$idx" -eq "$MASTER_RANK" ]]; then
            get_host_from_line "$line"
            return 0
        fi
        idx=$((idx + 1))
    done < <(read_host_lines "$RUN_HOSTFILE")
    return 1
}

find_master_log_file() {
    local master_host="$1"
    local candidates=""
    candidates=$(find "$LOG_DIR" -maxdepth 1 -type f \( \
        -iname "*${master_host}*" -o \
        -iname "*master*" -o \
        -iname "*rank0*" -o \
        -iname "*.log" \
    \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk '{print $2}')

    local file
    while read -r file; do
        [[ -z "$file" ]] && continue
        if grep -I -m1 -E 'step[^0-9]*[0-9]+|loss[^[:alpha:]]*[:= ]' "$file" >/dev/null 2>&1; then
            echo "$file"
            return 0
        fi
    done <<< "$candidates"

    echo "$candidates" | head -n 1
}

get_last_progress_ts() {
    local log_file="$1"
    python - "$log_file" <<'PY'
import os
import re
import sys
from datetime import datetime

path = sys.argv[1]
if not os.path.isfile(path):
    print(0)
    raise SystemExit(0)

patterns = [
    re.compile(r"step[^0-9]*\d+", re.IGNORECASE),
    re.compile(r"loss[^A-Za-z]*[:= ]", re.IGNORECASE),
]
ts_patterns = [
    re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"),
    re.compile(r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"),
]
last_ts = 0
found_progress = False

with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if not any(p.search(line) for p in patterns):
            continue
        found_progress = True
        parsed = None
        for tp in ts_patterns:
            m = tp.search(line)
            if not m:
                continue
            raw = m.group(1)
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S'):
                try:
                    parsed = int(datetime.strptime(raw, fmt).timestamp())
                    break
                except ValueError:
                    pass
            if parsed is not None:
                break
        if parsed is None:
            try:
                parsed = int(os.path.getmtime(path))
            except OSError:
                parsed = 0
        if parsed > last_ts:
            last_ts = parsed

if not found_progress:
    print(0)
else:
    print(last_ts)
PY
}

collect_target_pids_cmd='python - <<"PY"
import re
import subprocess
from collections import defaultdict

try:
    out = subprocess.check_output(["mthreads-gmi", "-pm"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    print("")
    raise SystemExit(0)

best = {}
for raw in out.splitlines():
    m = re.match(r"\s*(\d+)\s+(\d+)\s+([^\s]+)\s+([^\s]+)", raw)
    if not m:
        continue
    gpu = int(m.group(1))
    pid = m.group(2)
    mem = m.group(3)
    if mem.startswith("<"):
        mem_v = 0.0
    else:
        try:
            mem_v = float(mem.rstrip("%"))
        except ValueError:
            mem_v = 0.0
    cur = best.get(gpu)
    if cur is None or mem_v > cur[1]:
        best[gpu] = (pid, mem_v)

pids = []
seen = set()
for gpu in sorted(best):
    pid = best[gpu][0]
    if pid not in seen:
        seen.add(pid)
        pids.append(pid)
print(" ".join(pids))
PY'

dump_host_stacks() {
    local host="$1"
    local timestamp="$2"
    local remote_dir="${DUMP_BASE_DIR}/${timestamp}"
    local remote_cmd="mkdir -p '${remote_dir}' && pids=\$(${collect_target_pids_cmd}) && if [[ -z \"\$pids\" ]]; then echo NO_PID; exit 0; fi; for pid in \$pids; do xpu_timer_dump_driver --pid \"\$pid\" --dump-path '${remote_dir}' --pyspy --gdb; done"
    local ssh_out=""
    local ssh_rc=0
    ssh_out=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$host" "$remote_cmd" 2>&1) || ssh_rc=$?
    if [[ "$ssh_rc" -ne 0 ]]; then
        log "dump failed on ${host}, rc=${ssh_rc}, output=${ssh_out}"
        return 1
    fi
    log "dump finished on ${host}, remote_dir=${remote_dir}, output=${ssh_out}"
    return 0
}

trigger_dump_all_hosts() {
    local timestamp="$1"
    local pids=()
    local line host
    while read -r line; do
        [[ -z "$line" ]] && continue
        host=$(get_host_from_line "$line")
        (
            dump_host_stacks "$host" "$timestamp"
        ) &
        pids+=("$!")
    done < <(read_host_lines "$RUN_HOSTFILE")

    local rc=0
    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    return "$rc"
}

master_host=$(find_master_host || true)
if [[ -z "$master_host" ]]; then
    log "failed to determine master host from: $RUN_HOSTFILE"
    exit 1
fi
log "master host: $master_host"

hang_threshold_sec=$((HANG_MINUTES * 60))
hang_triggered=0

while true; do
    if [[ ! -d "$LOG_DIR" ]]; then
        log "log dir disappeared: $LOG_DIR"
        exit 1
    fi

    master_log=$(find_master_log_file "$master_host")
    if [[ -z "$master_log" || ! -f "$master_log" ]]; then
        log "master log file not found yet under: $LOG_DIR"
        sleep "$POLL_INTERVAL"
        continue
    fi

    last_progress_ts=$(get_last_progress_ts "$master_log")
    if [[ "$last_progress_ts" -le 0 ]]; then
        log "master log has no step/loss yet: $master_log"
        sleep "$POLL_INTERVAL"
        continue
    fi

    now_ts=$(date +%s)
    diff_sec=$((now_ts - last_progress_ts))
    log "master log: $master_log, last_progress_age=${diff_sec}s, threshold=${hang_threshold_sec}s"

    if [[ "$diff_sec" -gt "$hang_threshold_sec" ]]; then
        hang_triggered=1
        HANG_REASON="step_or_loss_silent_${diff_sec}s"
        dump_ts=$(date '+%Y%m%d_%H%M%S')
        log "hang detected, reason=${HANG_REASON}, triggering pyspy dump on all hosts"
        trigger_dump_all_hosts "$dump_ts" || true
        if [[ "$ALERT_ON_HANG" -eq 1 ]]; then
            echo "HANG_DETECTED reason=${HANG_REASON} master_log=${master_log} dump_ts=${dump_ts}"
        fi
        exit 100
    fi

    sleep "$POLL_INTERVAL"
done

