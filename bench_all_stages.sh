#!/usr/bin/env bash
# bench_all_stages.sh
# Sets up the AWS instance, builds and runs spira_bench for every stage branch,
# and saves all results to ~/bench_results/.
# Runs inside a tmux session so it survives SSH disconnect.
#
# PRE-REQUISITE: the flush_cache fix must be committed on every stage branch.
# Verify with:  for b in stage{1..9}/*; do git show $b:bench/spira_bench.cpp | grep -q flush_cache && echo "OK $b" || echo "MISSING $b"; done
#
# Usage:  chmod +x bench_all_stages.sh && ./bench_all_stages.sh
# Attach: tmux attach -t spira_bench
# Detach: Ctrl-B then D
# Log:    tail -f ~/bench_run.log

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$HOME/bench_results"
BUILD_DIR="$REPO_DIR/build_bench"
SESSION="spira_bench"

STAGES=(
    stage1/mvp-naive-implementation
    stage2/hash-buffer-and-algorithms
    stage3/simd-spmv-kernels
    stage4/crtp-open-locked-mode
    stage5/avx-scalar-gathering
    stage6/prefetch-and-hw-detection
    stage7/layout-aware-csr
    stage8/parallel-matrix-and-algorithms
    stage9/ilp-and-allocation-tuning
)

PARALLEL_STAGES=(
    stage8/parallel-matrix-and-algorithms
    stage9/ilp-and-allocation-tuning
)

# ---------------------------------------------------------------------------
# If not inside tmux, install tmux, then relaunch self inside a new session
# ---------------------------------------------------------------------------
if [ -z "${TMUX:-}" ]; then
    echo "[init] Installing tmux..."
    sudo apt-get update -qq
    sudo apt-get install -y tmux

    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -x 220 -y 50 \
        "bash $0 2>&1 | tee $HOME/bench_run.log; echo ''; echo '[done] All stages complete. Press enter to exit.'; read"

    echo "Benchmark session started in tmux."
    echo ""
    echo "  Attach:   tmux attach -t $SESSION"
    echo "  Detach:   Ctrl-B then D"
    echo "  Log:      tail -f ~/bench_run.log"
    exit 0
fi

# ---------------------------------------------------------------------------
# From here we are running inside tmux
# ---------------------------------------------------------------------------

is_parallel_stage() {
    local stage="$1"
    for p in "${PARALLEL_STAGES[@]}"; do
        [[ "$stage" == "$p" ]] && return 0
    done
    return 1
}

setup_system() {
    echo "[setup] Installing build dependencies..."
    sudo apt-get update -qq
    # Install g++-13 explicitly — g++ alias on Ubuntu 22.04 is g++-11
    # which has incomplete C++23 support. g++-13 is fully sufficient.
    sudo apt-get install -y cmake ninja-build g++-13 git numactl

    # Make g++-13 the default
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 60
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60
    echo "[setup] Compiler: $(g++ --version | head -1)"

    echo "[setup] Setting CPU governor to performance..."
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -f "$cpu" ] && echo performance | sudo tee "$cpu" > /dev/null
    done

    echo "[setup] Disabling Intel Turbo Boost..."
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
    fi

    echo "[setup] Disabling NUMA balancing..."
    echo 0 | sudo tee /proc/sys/kernel/numa_balancing > /dev/null

    echo "[setup] Disabling transparent huge pages..."
    echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null

    echo "[setup] Reducing swappiness..."
    echo 1 | sudo tee /proc/sys/vm/swappiness > /dev/null

    # Move IRQ handlers off core 0 so single-threaded benchmarks are not
    # interrupted. Set all IRQ affinities to use cores 1+ (bitmask 0xfffe
    # = all cores except 0).
    echo "[setup] Moving IRQs off core 0..."
    for irq_aff in /proc/irq/*/smp_affinity; do
        echo fe | sudo tee "$irq_aff" > /dev/null 2>&1 || true
    done

    echo "[setup] System topology:"
    lscpu | grep -E "NUMA|Socket|Core\(s\)|Thread|^CPU\(s\)"
    echo ""
    numactl --hardware
    echo ""
    echo "[setup] Done."
}

verify_flush_cache() {
    local stage="$1"
    if ! git -C "$REPO_DIR" show "${stage}:bench/spira_bench.cpp" 2>/dev/null | grep -q "flush_cache"; then
        echo ""
        echo "ERROR: flush_cache fix is not committed on branch '$stage'."
        echo "       Commit the fix to that branch before running this script."
        echo ""
        exit 1
    fi
}

build_stage() {
    local stage="$1"
    echo "[build] $stage"
    rm -rf "$BUILD_DIR"
    cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPIRA_BUILD_BENCHMARKS=ON \
        -GNinja
    cmake --build "$BUILD_DIR" --target spira_bench -j"$(nproc)"
}

run_stage() {
    local stage="$1"
    local safe_name="${stage//\//_}"
    local out_file="$RESULTS_DIR/${safe_name}.json"

    mkdir -p "$RESULTS_DIR"

    local bench_args=(
        --benchmark_out="$out_file"
        --benchmark_out_format=json
        --benchmark_repetitions=3
        --benchmark_report_aggregates_only=true
        --benchmark_min_time=0.1s
        --benchmark_color=false
    )

    if is_parallel_stage "$stage"; then
        # Allow all cores on NUMA node 0 so threads can spread naturally.
        # membind=0 keeps allocations local to avoid cross-NUMA latency.
        echo "[run] $stage → $out_file  (multi-threaded: all cores, NUMA node 0)"
        numactl --cpunodebind=0 --membind=0 \
            "$BUILD_DIR/spira_bench" "${bench_args[@]}"
    else
        # Pin to core 0 only. IRQs have been moved off core 0 above.
        echo "[run] $stage → $out_file  (single-threaded: core 0, NUMA node 0)"
        taskset -c 0 numactl --membind=0 \
            "$BUILD_DIR/spira_bench" "${bench_args[@]}"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CURRENT_BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)

echo "============================================"
echo "  spira benchmark run — $(date)"
echo "  results → $RESULTS_DIR"
echo "============================================"
echo ""

# Verify flush_cache is committed on every branch before we start
echo "[preflight] Checking flush_cache fix on all branches..."
for stage in "${STAGES[@]}"; do
    verify_flush_cache "$stage"
    echo "  OK  $stage"
done
echo ""

setup_system

for stage in "${STAGES[@]}"; do
    echo ""
    echo "=========================================="
    echo "  STAGE: $stage"
    echo "=========================================="

    git -C "$REPO_DIR" checkout "$stage"
    build_stage "$stage"
    run_stage "$stage"
done

git -C "$REPO_DIR" checkout "$CURRENT_BRANCH"

echo ""
echo "============================================"
echo "  All stages complete — $(date)"
echo "  Results:"
ls -lh "$RESULTS_DIR"
echo "============================================"
