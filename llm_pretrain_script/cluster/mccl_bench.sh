#!/bin/bash
# Usage: ./mccl_bench.sh HOSTFILE [--worldsize N] [--netcheck]
# HOSTFILE: Path to the hostfile
# --worldsize N: Split hostfile into groups of N nodes each and run tests in parallel
# --netcheck: Network check mode - test pairs of nodes and identify problematic IPs

HOSTFILE="$1"
WORLDSIZE=""
NETCHECK=0
# Minimum average bandwidth for netcheck pass (GB/s). Below this will be treated as fail.
NETCHECK_AVG_BW_MIN="${NETCHECK_AVG_BW_MIN:-170}"
shift

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
        case $1 in
                --worldsize)
                        WORLDSIZE="$2"
                        shift 2
                        ;;
                --netcheck)
                        NETCHECK=1
                        shift
                        ;;
                *)
                        echo "Unknown option: $1"
                        echo "Usage: $0 HOSTFILE [--worldsize N] [--netcheck]"
                        exit 1
                        ;;
        esac
done

# Function to run mpirun with a given hostfile
run_mpi_test() {
        local hostfile="$1"
        local group_id="$2"
        local node_count=$(wc -l < "$hostfile")
        local timeout_sec="${MPI_TIMEOUT:-90}"
        # Calculate total process count from slots (sum of all slots=N values)
        local total_slots=$(awk -F'slots=' '{sum += $2} END {print sum}' "$hostfile")
        # If no slots specified, fall back to node count
        if [[ -z "$total_slots" ]] || [[ "$total_slots" -eq 0 ]]; then
                total_slots=$node_count
        fi
        local output_file="${LOG_DIR}/mccl_allreduce_group${group_id}.txt"
        if [[ $NETCHECK -eq 1 ]]; then
                local group_ips=$(awk '{print $1}' "$hostfile" | paste -sd '_' - | sed 's/[^A-Za-z0-9_.-]/_/g')
                output_file="${LOG_DIR}/mccl_allreduce_group${group_id}_${group_ips}.txt"
                echo "[Group $group_id] IPs: $(awk '{print $1}' "$hostfile" | paste -sd ' ' -)"
                echo "[Group $group_id] Log file: $output_file"
        fi

        timeout "$timeout_sec" /usr/local/openmpi/bin/mpirun -hostfile "$hostfile" \
                -np "$total_slots" \
                --allow-run-as-root \
                --timeout 90 \
                --verbose \
                -mca plm_base_verbose 5 \
                -mca orte_abort_timeout 300 \
                -mca btl_tcp_if_include bond0 \
                --prtemca prte_keep_fqdn_hostnames 1 \
                --bind-to none \
                -x LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/lib/:/usr/lib64:/usr/local/musa/lib:/usr/local/musa/mudnn/lib:/usr/local/openmpi/lib \
                -x PATH=/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/musa/mudnn/mudnn_bench/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
                -x MCCL_PROTOS=2 \
                -x MCCL_ALGOS=1 \
                -x MCCL_CROSS_NIC=1 \
                -x MCCL_IB_TIMEOUT=20 \
                -x MCCL_IB_RETRY_CNT=7 \
        /usr/local/musa/mccl_test/all_reduce_perf -n 3 -b 512M -e 20480M -f 2 -g 1 -t 1 > "$output_file" 2>&1

        # /usr/local/musa/mccl_test/all_gather_perf -n 3 -b 800M -e 2048M -f 2 -g 1 -t 1 > "$output_file" 2>&1
        # /usr/local/musa/mccl_test/all_reduce_perf -n 3 -b 512M -e 20480M -f 2 -g 1 -t 1 > "$output_file" 2>&1
        #        -x MCCL_DEBUG=INFO \
        #        -x MCCL_DEBUG_SUBSYS=ALL \
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            echo "[Group $group_id] Completed"
        elif [[ $exit_code -eq 124 ]]; then
            echo "[Group $group_id] Timeout after ${timeout_sec}s"
            echo "ERROR: Timeout after ${timeout_sec}s" >> "$output_file"
            echo "Hosts in group: $(awk '{print $1}' "$hostfile" | tr '\n' ' ')" >> "$output_file"
            echo "Problematic IPs (timeout):"
            awk '{print "  - "$1}' "$hostfile"
        else
            echo "[Group $group_id] Completed with exit code $exit_code"
        fi

#       -x MUSA_VISIBLE_DEVICES=3 \
}

# Check if hostfile exists
if [[ ! -f "$HOSTFILE" ]]; then
    echo "Error: Hostfile '$HOSTFILE' not found!"
    exit 1
fi

# Create log directory with mccl prefix and timestamp
LOG_DIR="mccl_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

total_lines=$(wc -l < "$HOSTFILE")
echo "Total nodes in hostfile: $total_lines"

# Function to check if a log file has errors
check_group_log() {
    local logfile="$1"
    if [[ ! -f "$logfile" ]]; then
        return 1
    fi
    # Check for error keywords
    if grep -iq "error\|fail\|abort\|exception" "$logfile" 2>/dev/null; then
        return 1
    fi
    # Check for non-zero #wrong values
    if awk '$1 ~ /^[0-9]+$/ && NF >= 13 && $9 ~ /^[0-9]+$/ && $9 != "0" {exit 1}' "$logfile" 2>/dev/null; then
        if awk '$1 ~ /^[0-9]+$/ && NF >= 13 && $13 ~ /^[0-9]+$/ && $13 != "0" {exit 1}' "$logfile" 2>/dev/null; then
            :
        else
            return 1
        fi
    else
        return 1
    fi

    # Check average busbw (out-of-place, column 8). Below threshold => fail
    avg_bw=$(awk '$1 ~ /^[0-9]+$/ && NF >= 13 {sum += $8; cnt++} END {if (cnt>0) printf "%.3f", sum/cnt; else print ""}' "$logfile" 2>/dev/null)
    if [[ -z "$avg_bw" ]]; then
        return 1
    fi
    is_low=$(awk -v bw="$avg_bw" -v min="$NETCHECK_AVG_BW_MIN" 'BEGIN {print (bw < min) ? 1 : 0}')
    if [[ "$is_low" -eq 1 ]]; then
        return 1
    fi

    return 0
}

# Network check mode
if [[ $NETCHECK -eq 1 ]]; then
    echo "=== Network Check Mode ==="

    # Create temporary directory
    tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT

    # Read all hosts into array
    mapfile -t all_hosts < "$HOSTFILE"
    num_hosts=${#all_hosts[@]}

    # Build IP map to preserve slots info
    declare -A ip_map
    for host_line in "${all_hosts[@]}"; do
        ip=$(echo "$host_line" | awk '{print $1}')
        ip_map[$ip]="$host_line"
    done

    if [[ $num_hosts -lt 2 ]]; then
        echo "Error: Need at least 2 hosts for network check"
        exit 1
    fi

    echo "Phase 1: Testing pairs of nodes ($((num_hosts / 2)) pairs)..."

    # Phase 1: Test pairs (2 nodes each)
    declare -A phase1_results   # group_id -> "pass" or "fail"
    declare -A phase1_ips       # group_id -> "ip1 ip2"
    declare -A phase1_logfiles  # group_id -> logfile path
    group_id=0
    pids=()

    # 记录最后一个未配对的节点（奇数情况）
    odd_host=""
    odd_ip=""

    for ((i=0; i<num_hosts-1; i+=2)); do
        group_id=$((group_id + 1))
        host1="${all_hosts[$i]}"
        host2="${all_hosts[$((i+1))]}"

        # Extract only IP address (first field before space or slots=)
        ip1=$(echo "$host1" | awk '{print $1}')
        ip2=$(echo "$host2" | awk '{print $1}')

        # Create hostfile for this pair
        pair_hostfile="$tmp_dir/pair_${group_id}.hostfile"
        echo "$host1" > "$pair_hostfile"
        echo "$host2" >> "$pair_hostfile"

        phase1_ips[$group_id]="$ip1 $ip2"
        phase1_logfiles[$group_id]="${LOG_DIR}/mccl_allreduce_groupp1_${group_id}_${ip1}_${ip2}.txt"
        echo "  [Phase 1 Group $group_id] IPs: $ip1 $ip2"

        run_mpi_test "$pair_hostfile" "p1_${group_id}" &
        pids+=($!)
    done

    # Handle odd number of hosts - 记录最后一个未配对的节点
    if [[ $((num_hosts % 2)) -eq 1 ]]; then
        odd_host="${all_hosts[$((num_hosts-1))]}"
        odd_ip=$(echo "$odd_host" | awk '{print $1}')
        echo "Warning: Odd number of hosts, last host ($odd_ip) will be tested in phase 2"
    fi

    # Wait for phase 1
    echo "Running ${#pids[@]} pair tests in parallel..."
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    # Analyze phase 1 results
    echo "Analyzing phase 1 results..."
    failed_ips=()
    passed_ips=()

    for gid in "${!phase1_ips[@]}"; do
        logfile="${phase1_logfiles[$gid]}"
        ips=(${phase1_ips[$gid]})

        if check_group_log "$logfile"; then
            phase1_results[$gid]="pass"
            for ip in "${ips[@]}"; do
                passed_ips+=("$ip")
            done
        else
            phase1_results[$gid]="fail"
            for ip in "${ips[@]}"; do
                failed_ips+=("$ip")
            done
            echo "  [Phase 1 FAIL] Group $gid: ${ips[*]}"
        fi
    done

    echo "Phase 1 complete: ${#passed_ips[@]} passed IPs, ${#failed_ips[@]} failed IPs"

    # 将奇数节点加入待测试列表（如果有的话）
    untested_ips=()
    if [[ -n "$odd_ip" ]]; then
        untested_ips+=("$odd_ip")
        echo "Note: Odd host ($odd_ip) added to phase 2 testing"
    fi

    # Phase 2: Cross-test failed IPs with passed IPs
    if [[ ${#failed_ips[@]} -eq 0 && ${#untested_ips[@]} -eq 0 ]]; then
        echo "All pairs passed! No network issues detected."
        exit 0
    fi

    if [[ ${#passed_ips[@]} -eq 0 ]]; then
        echo "All pairs failed! Cannot determine which specific IPs are problematic."
        echo "Failed IPs: ${failed_ips[*]}"
        exit 1
    fi

    # 合并 failed_ips 和 untested_ips 进行 Phase 2 测试
    phase2_test_ips=("${failed_ips[@]}" "${untested_ips[@]}")

    echo ""
    echo "Phase 2: Cross-testing ${#phase2_test_ips[@]} IPs with passed IPs..."

    declare -A phase2_results   # failed_ip -> "pass" or "fail"
    declare -A phase2_logfiles  # group_id -> logfile path
    declare -A phase2_ips       # group_id -> "test_ip reference_ip"
    pids=()
    group_id=0

    # Use available passed IPs as references
    num_passed=${#passed_ips[@]}
    echo "Using $num_passed passed IPs as references for cross-testing"

    failed_idx=0
    for test_ip in "${phase2_test_ips[@]}"; do
        group_id=$((group_id + 1))

        # Select partner from passed_ips in round-robin fashion
        partner_idx=$((failed_idx % num_passed))
        reference_ip="${passed_ips[$partner_idx]}"

        # Create hostfile for cross-test
        cross_hostfile="$tmp_dir/cross_${group_id}.hostfile"
        # Use ip_map to preserve slots info
        echo "${ip_map[$test_ip]}" > "$cross_hostfile"
        echo "${ip_map[$reference_ip]}" >> "$cross_hostfile"

        phase2_ips[$group_id]="$test_ip $reference_ip"
        phase2_logfiles[$group_id]="${LOG_DIR}/mccl_allreduce_groupp2_${group_id}_${test_ip}_${reference_ip}.txt"
        echo "  [Phase 2 Group $group_id] IPs: $test_ip $reference_ip"

        run_mpi_test "$cross_hostfile" "p2_${group_id}" &
        pids+=($!)

        # Store mapping
        eval "phase2_map_$group_id=\"$test_ip\""
        failed_idx=$((failed_idx + 1))
    done

    # Wait for phase 2
    echo "Running ${#pids[@]} cross-tests in parallel..."
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    # Analyze phase 2 results
    echo "Analyzing phase 2 results..."
    confirmed_bad_ips=()
    confirmed_good_ips=()

    for ((gid=1; gid<=group_id; gid++)); do
        eval "tested_ip=\${phase2_map_$gid}"
        logfile="${phase2_logfiles[$gid]}"

        if check_group_log "$logfile"; then
            phase2_results[$tested_ip]="pass"
            confirmed_good_ips+=("$tested_ip")
            echo "  [Phase 2 PASS] $tested_ip - likely partner issue in phase 1 or untested odd host OK"
        else
            phase2_results[$tested_ip]="fail"
            confirmed_bad_ips+=("$tested_ip")
            echo "  [Phase 2 FAIL] $tested_ip - confirmed problematic"
        fi
    done

    echo ""
    echo "========================================="
    echo "Network Check Summary"
    echo "========================================="
    echo "Total hosts tested: $num_hosts"
    echo "Phase 1 pairs tested: $((num_hosts / 2))"
    echo "Phase 1 failed pairs: $((${#failed_ips[@]} / 2))"
    if [[ -n "$odd_ip" ]]; then
        echo "Odd host (tested in phase 2): $odd_ip"
    fi
    echo "Phase 2 tested IPs: ${#phase2_test_ips[@]}"
    echo "Phase 2 confirmed bad IPs: ${#confirmed_bad_ips[@]}"
    echo "Phase 2 confirmed good IPs: ${#confirmed_good_ips[@]}"

    if [[ ${#confirmed_bad_ips[@]} -gt 0 ]]; then
        echo ""
        echo "Problematic IPs (failed phase 2):"
        for ip in "${confirmed_bad_ips[@]}"; do
            echo "  - $ip"
        done
    else
        echo ""
        echo "No single IP confirmed as problematic."
        echo "Phase 1 failures may be due to pair-specific issues."
    fi

    exit 0
fi

if [[ -z "$WORLDSIZE" ]]; then
    # No worldsize specified, run with original hostfile
    echo "No worldsize specified, running with all nodes..."
    node_count=$total_lines
    run_mpi_test "$HOSTFILE" "all"
else
    # Validate worldsize
    if ! [[ "$WORLDSIZE" =~ ^[0-9]+$ ]] || [[ "$WORLDSIZE" -lt 1 ]]; then
        echo "Error: worldsize must be a positive integer"
        exit 1
    fi

    if [[ $((total_lines % WORLDSIZE)) -ne 0 ]]; then
        echo "Warning: Total nodes ($total_lines) is not evenly divisible by worldsize ($WORLDSIZE)"
        echo "Some groups may have fewer nodes"
    fi

    num_groups=$(( (total_lines + WORLDSIZE - 1) / WORLDSIZE ))
    echo "Splitting hostfile into $num_groups groups of $WORLDSIZE nodes each..."

    # Create temporary directory for split hostfiles
    tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT

    # Split hostfile into groups
    split -l "$WORLDSIZE" -d --additional-suffix=.hostfile "$HOSTFILE" "$tmp_dir/group_"

    # Run MPI tests for each group in parallel
    group_id=0
    pids=()
    declare -A group_hostfiles
    declare -A group_logfiles
    for split_hostfile in "$tmp_dir"/group_*.hostfile; do
        if [[ -f "$split_hostfile" ]]; then
            group_id=$((group_id + 1))
            group_hostfiles[$group_id]="$split_hostfile"
            group_logfiles[$group_id]="${LOG_DIR}/mccl_allreduce_group${group_id}.txt"
            run_mpi_test "$split_hostfile" "$group_id" &
            pids+=($!)
        fi
    done

    # Wait for all parallel jobs to complete
    echo "Running $group_id groups in parallel..."
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    # Check logs for errors and report
    echo "Checking results..."
    error_found=0
    declare -a error_group_msgs      # 出错的group id列表
    declare -A error_group_hosts     # group id -> hosts列表
    declare -A error_group_details   # group id -> 错误详情

    # First pass: collect busbw data from all groups for comparison
    declare -A oop_busbw_by_size  # size -> "gid1:busbw1 gid2:busbw2 ..."
    declare -A ip_busbw_by_size   # size -> "gid1:busbw1 gid2:busbw2 ..."

    for gid in "${!group_hostfiles[@]}"; do
        logfile="${LOG_DIR}/mccl_allreduce_group${gid}.txt"
        if [[ -f "$logfile" ]]; then
            # Extract size, out-of-place busbw (col 8), in-place busbw (col 12)
            # Only match lines where first column is a number (data lines)
            while read -r size oop_bw ip_bw; do
                if [[ -n "$size" ]]; then
                    oop_busbw_by_size[$size]+="$gid:$oop_bw "
                    ip_busbw_by_size[$size]+="$gid:$ip_bw "
                fi
            done < <(awk '$1 ~ /^[0-9]+$/ && NF >= 13 {print $1, $8, $12}' "$logfile" 2>/dev/null)
        fi
    done

    # Second pass: check for errors and performance anomalies
    for gid in "${!group_hostfiles[@]}"; do
        logfile="${LOG_DIR}/mccl_allreduce_group${gid}.txt"
        if [[ -f "$logfile" ]]; then
                has_error=0
                error_msgs=""
                group_ips=()
                # Collect IPs in this group
                while read -r ipline; do
                    ipaddr=$(echo "$ipline" | awk '{print $1}')
                    group_ips+=("$ipaddr")
                done < "${group_hostfiles[$gid]}"

                # Check if bandwidth output exists (data lines with busbw)
                busbw_lines=$(awk '$1 ~ /^[0-9]+$/ && NF >= 13 {count++} END {print count+0}' "$logfile" 2>/dev/null)
                if [[ "$busbw_lines" -eq 0 ]]; then
                    has_error=1
                    error_msgs+="[No Bandwidth Output]\n"
                    error_msgs+="  No valid busbw data found in log file\n"
                    # Capture last 20 lines for debugging
                    error_msgs+="  Last 20 lines of log:\n"
                    error_msgs+=$(tail -20 "$logfile" | sed 's/^/    /')
                    error_msgs+="\n"
                fi

                # Check for error keywords (case-insensitive)
                if grep -iq "error\|fail\|abort\|exception" "$logfile" 2>/dev/null; then
                    has_error=1
                    error_msgs+="[Keyword Errors]\n"
                    error_msgs+=$(grep -in "error\|fail\|abort\|exception" "$logfile" | head -20)
                    error_msgs+="\n"
                fi

                # Check for non-zero #wrong values in performance output
                wrong_oop=$(awk '$1 ~ /^[0-9]+$/ && NF >= 13 && $9 ~ /^[0-9]+$/ && $9 != "0" {print "  size="$1" #wrong="$9}' "$logfile" 2>/dev/null)
                wrong_ip=$(awk '$1 ~ /^[0-9]+$/ && NF >= 13 && $13 ~ /^[0-9]+$/ && $13 != "0" {print "  size="$1" #wrong="$13}' "$logfile" 2>/dev/null)
                if [[ -n "$wrong_oop" ]]; then
                    has_error=1
                    error_msgs+="[Out-of-place #wrong non-zero]\n"
                    error_msgs+="$wrong_oop\n"
                fi
                if [[ -n "$wrong_ip" ]]; then
                    has_error=1
                    error_msgs+="[In-place #wrong non-zero]\n"
                    error_msgs+="$wrong_ip\n"
                fi

                # Check busbw performance anomalies (only when more than 1 group)
                num_total_groups=${#group_hostfiles[@]}
                if [[ $num_total_groups -gt 1 ]]; then
                    oop_slow=""
                    ip_slow=""
                    while read -r size oop_bw ip_bw; do
                        if [[ -n "$size" ]]; then
                            oop_data="${oop_busbw_by_size[$size]}"
                            if [[ -n "$oop_data" ]]; then
                                max_oop=$(echo "$oop_data" | tr ' ' '\n' | grep -v '^$' | cut -d: -f2 | sort -n | tail -1)
                                if [[ -n "$max_oop" ]]; then
                                    is_slow=$(awk -v bw="$oop_bw" -v max="$max_oop" 'BEGIN {print (bw < max * 0.7) ? 1 : 0}')
                                    if [[ "$is_slow" -eq 1 ]]; then
                                        diff_pct=$(awk -v bw="$oop_bw" -v max="$max_oop" 'BEGIN {printf "%.1f", (1 - bw / max) * 100}')
                                        oop_slow+="  size=$size busbw=$oop_bw (max=$max_oop, diff=${diff_pct}%)\n"
                                    fi
                                fi
                            fi
                            ip_data="${ip_busbw_by_size[$size]}"
                            if [[ -n "$ip_data" ]]; then
                                max_ip=$(echo "$ip_data" | tr ' ' '\n' | grep -v '^$' | cut -d: -f2 | sort -n | tail -1)
                                if [[ -n "$max_ip" ]]; then
                                    is_slow=$(awk -v bw="$ip_bw" -v max="$max_ip" 'BEGIN {print (bw < max * 0.7) ? 1 : 0}')
                                    if [[ "$is_slow" -eq 1 ]]; then
                                        diff_pct=$(awk -v bw="$ip_bw" -v max="$max_ip" 'BEGIN {printf "%.1f", (1 - bw / max) * 100}')
                                        ip_slow+="  size=$size busbw=$ip_bw (max=$max_ip, diff=${diff_pct}%)\n"
                                    fi
                                fi
                            fi
                        fi
                    done < <(awk '$1 ~ /^[0-9]+$/ && NF >= 13 {print $1, $8, $12}' "$logfile" 2>/dev/null)
                    if [[ -n "$oop_slow" ]]; then
                        has_error=1
                        error_msgs+="[Out-of-place busbw >30% slower than max]\n"
                        error_msgs+="$oop_slow"
                    fi
                    if [[ -n "$ip_slow" ]]; then
                        has_error=1
                        error_msgs+="[In-place busbw >30% slower than max]\n"
                        error_msgs+="$ip_slow"
                    fi
                fi

                # 收集出错IP和错误信息
                if [[ $has_error -eq 1 ]]; then
                    error_found=1
                    group_error_info="[Group $gid] Hosts: "
                    host_list=""
                    for ip in "${group_ips[@]}"; do
                        # 转换为worker主机名格式，数字部分5位
                        # 例：10.121.32.1 -> worker32001
                        if [[ $ip =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
                            num=$(printf '%d%03d' ${BASH_REMATCH[3]} ${BASH_REMATCH[4]})
                            worker_name="worker$(printf '%05d' $num)"
                            error_hosts+=("$worker_name($ip)")
                            host_list+="$worker_name($ip) "
                        else
                            error_hosts+=("$ip")
                            host_list+="$ip "
                        fi
                    done
                    # 保存该group的错误信息
                    error_group_msgs+=("$gid")
                    error_group_hosts["$gid"]="$host_list"
                    error_group_details["$gid"]="$error_msgs"
                fi
        fi
    done

    if [[ $error_found -eq 0 ]]; then
        echo "No errors found in any group logs."
    else
        echo ""
        echo "=========================================="
        echo "Summary: Errors detected in some groups!"
        echo "=========================================="

        # 输出每个出错group的详细信息
        for gid in "${error_group_msgs[@]}"; do
            echo ""
            echo "[Group $gid] ERROR"
            echo "  Hosts: ${error_group_hosts[$gid]}"
            echo "  Error details:"
            echo -e "${error_group_details[$gid]}" | sed 's/^/    /'
        done

        echo ""
        echo "------------------------------------------"
        # 集中输出所有出错的worker主机名和IP
        if [[ ${#error_hosts[@]} -gt 0 ]]; then
            echo "All failed hosts:"
            for h in "${error_hosts[@]}"; do
                echo "  $h"
            done
            echo "Total failed nodes: ${#error_hosts[@]}"
        fi
    fi
fi
