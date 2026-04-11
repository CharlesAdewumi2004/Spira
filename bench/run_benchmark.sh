#!/usr/bin/env bash
# =============================================================================
# Spira benchmark runner — AWS r8i.4xlarge (or similar)
#
# Builds all 4 stages in git worktrees, then launches ONE tmux window that
# runs all stages sequentially (stage1 → stage2 → stage3 → stage4).
# Single-threaded stages (1-3) are pinned to one physical core; stage 4
# (parallel) gets all physical cores, HT optionally disabled.
#
# Usage:
#   chmod +x bench/run_benchmark.sh
#   ./bench/run_benchmark.sh [--no-smt-disable]
#
# Results → bench_results/results_<timestamp>.txt  (console)
#           bench_results/results_<timestamp>_<stage>.json  (machine-readable)
# Retrieve:
#   scp ubuntu@<ip>:~/spira/bench_results/results_<timestamp>* .
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.txt"
WORKTREES_DIR="${REPO_ROOT}/.bench_worktrees"
SESSION="spira_bench"
RUN_ALL="${WORKTREES_DIR}/run_all.sh"

declare -A STAGE_BRANCH=(
    [stage1]="stage1/MVP"
    [stage2]="stage2/SIMD-Kernels"
    [stage3]="stage3/layout-aware-csr"
    [stage4]="stage4/MutilThreaded"
)
STAGE_ORDER=(stage1 stage2 stage3 stage4)

DISABLE_SMT=true
for arg in "$@"; do
    [[ "$arg" == "--no-smt-disable" ]] && DISABLE_SMT=false
done

# =============================================================================
# 1. Detect CPU topology
# =============================================================================
echo "=== Detecting CPU topology ==="

declare -A PHYS_TO_CPU
while IFS=',' read -r cpu core; do
    [[ "$cpu" =~ ^[0-9]+$ ]] || continue
    [[ -z "${PHYS_TO_CPU[$core]+_}" ]] && PHYS_TO_CPU["$core"]="$cpu"
done < <(lscpu -p=cpu,core)

PHYSICAL_CPUS=()
for core in $(printf '%s\n' "${!PHYS_TO_CPU[@]}" | sort -n); do
    PHYSICAL_CPUS+=("${PHYS_TO_CPU[$core]}")
done

NUM_PHYSICAL="${#PHYSICAL_CPUS[@]}"
SINGLE_CORE="${PHYSICAL_CPUS[0]}"
PARALLEL_CPUS="$(IFS=','; echo "${PHYSICAL_CPUS[*]}")"

echo "  Physical cores : ${NUM_PHYSICAL}"
echo "  Single-core pin: CPU ${SINGLE_CORE}"
echo "  Parallel CPUs  : ${PARALLEL_CPUS}"

# =============================================================================
# 2. Optionally disable SMT (requires root)
# =============================================================================
SMT_WAS_ON=false
SMT_CTL="/sys/devices/system/cpu/smt/control"

restore_smt() {
    if [[ "$SMT_WAS_ON" == true && -f "$SMT_CTL" && $EUID -eq 0 ]]; then
        echo "Restoring SMT..."
        echo on > "$SMT_CTL"
    fi
}
trap restore_smt EXIT

if [[ "$DISABLE_SMT" == true && -f "$SMT_CTL" ]]; then
    if [[ "$(cat "$SMT_CTL")" != "off" ]]; then
        if [[ $EUID -eq 0 ]]; then
            echo "=== Disabling SMT/HT ==="
            echo off > "$SMT_CTL"
            SMT_WAS_ON=true

            # Re-read CPU list now that HT siblings are offline
            unset PHYS_TO_CPU
            declare -A PHYS_TO_CPU
            while IFS=',' read -r cpu core; do
                [[ "$cpu" =~ ^[0-9]+$ ]] || continue
                [[ -z "${PHYS_TO_CPU[$core]+_}" ]] && PHYS_TO_CPU["$core"]="$cpu"
            done < <(lscpu -p=cpu,core)
            PHYSICAL_CPUS=()
            for core in $(printf '%s\n' "${!PHYS_TO_CPU[@]}" | sort -n); do
                PHYSICAL_CPUS+=("${PHYS_TO_CPU[$core]}")
            done
            NUM_PHYSICAL="${#PHYSICAL_CPUS[@]}"
            SINGLE_CORE="${PHYSICAL_CPUS[0]}"
            PARALLEL_CPUS="$(IFS=','; echo "${PHYSICAL_CPUS[*]}")"
            echo "  SMT off — parallel CPUs: ${PARALLEL_CPUS}"
        else
            echo "WARNING: not root, cannot disable SMT (pass --no-smt-disable to silence)"
        fi
    else
        echo "=== SMT already off ==="
    fi
fi

# =============================================================================
# 3. Build each stage in its own worktree
# =============================================================================
mkdir -p "${RESULTS_DIR}" "${WORKTREES_DIR}"

echo ""
echo "=== Building stages ==="

declare -A BENCH_BIN

for stage in "${STAGE_ORDER[@]}"; do
    branch="${STAGE_BRANCH[$stage]}"
    wt="${WORKTREES_DIR}/${stage}"
    bld="${wt}/build_bench"
    bin="${bld}/spira_bench"

    echo ""
    echo "--- ${stage} (${branch}) ---"

    if [[ ! -d "$wt" ]]; then
        git -C "${REPO_ROOT}" worktree add "${wt}" "${branch}"
    else
        echo "  worktree exists — pulling"
        git -C "${wt}" pull --ff-only 2>/dev/null || true
    fi

    mkdir -p "${bld}"

    cmake -S "${wt}" -B "${bld}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPIRA_BUILD_TESTS=OFF \
        -DSPIRA_BUILD_BENCHMARKS=ON \
        -DSPIRA_ENABLE_SIMD=ON 2>&1 | tail -3

    cmake --build "${bld}" --target spira_bench -j"$(nproc)" 2>&1 | tail -5

    if [[ ! -x "$bin" ]]; then
        echo "ERROR: binary not found: ${bin}" >&2
        exit 1
    fi

    BENCH_BIN["$stage"]="${bin}"
    echo "  ok: ${bin}"
done

# =============================================================================
# 4. Write a single sequential runner script (no nested heredoc)
# =============================================================================
echo ""
echo "=== Writing runner script ==="

printf '#!/usr/bin/env bash\n' > "${RUN_ALL}"
printf 'set -euo pipefail\n\n' >> "${RUN_ALL}"
printf 'RESULTS_FILE="%s"\n\n' "${RESULTS_FILE}" >> "${RUN_ALL}"

for stage in "${STAGE_ORDER[@]}"; do
    bin="${BENCH_BIN[$stage]}"
    json="${RESULTS_FILE%.txt}_${stage}.json"

    if [[ "$stage" == "stage4" ]]; then
        cpus="${PARALLEL_CPUS}"
        label="CPUs ${PARALLEL_CPUS} (${NUM_PHYSICAL} physical cores, parallel)"
    else
        cpus="${SINGLE_CORE}"
        label="CPU ${SINGLE_CORE} (single physical core, serial)"
    fi

    printf 'echo ""\n' >> "${RUN_ALL}"
    printf 'echo "============================================================"\n' >> "${RUN_ALL}"
    printf 'echo "  %s — $(date)"\n' "${stage}" >> "${RUN_ALL}"
    printf 'echo "  Pin : %s"\n' "${label}" >> "${RUN_ALL}"
    printf 'echo "============================================================"\n' >> "${RUN_ALL}"
    printf '{ echo ""; echo "============================================================"; echo "  %s — $(date)"; echo "  Pin : %s"; echo "============================================================"; } >> "${RESULTS_FILE}"\n' \
        "${stage}" "${label}" >> "${RUN_ALL}"
    printf 'taskset -c %s \\\n' "${cpus}" >> "${RUN_ALL}"
    printf '    "%s" \\\n' "${bin}" >> "${RUN_ALL}"
    printf '    --benchmark_format=console \\\n' >> "${RUN_ALL}"
    printf '    --benchmark_repetitions=5 \\\n' >> "${RUN_ALL}"
    printf '    --benchmark_report_aggregates_only=true \\\n' >> "${RUN_ALL}"
    printf '    --benchmark_out="%s" \\\n' "${json}" >> "${RUN_ALL}"
    printf '    --benchmark_out_format=json \\\n' >> "${RUN_ALL}"
    printf '    2>&1 | tee -a "${RESULTS_FILE}"\n' >> "${RUN_ALL}"
    printf 'echo "  %s done. JSON → %s"\n\n' "${stage}" "${json}" >> "${RUN_ALL}"
done

printf 'echo ""\n' >> "${RUN_ALL}"
printf 'echo "All stages complete."\n' >> "${RUN_ALL}"
printf 'echo "Results: ${RESULTS_FILE}"\n' >> "${RUN_ALL}"

chmod +x "${RUN_ALL}"

# =============================================================================
# 5. Write results header
# =============================================================================
{
    echo "Spira Benchmark Run — ${TIMESTAMP}"
    echo "Host    : $(hostname)"
    echo "Kernel  : $(uname -r)"
    echo "CPU     : $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//')"
    echo "Cores   : ${NUM_PHYSICAL} physical, parallel mask ${PARALLEL_CPUS}"
    echo "SMT off : ${SMT_WAS_ON}"
    echo "--------------------------------------------------------------"
} > "${RESULTS_FILE}"

# =============================================================================
# 6. Launch tmux — single window, all stages sequential
# =============================================================================
echo ""
echo "=== Launching tmux session '${SESSION}' ==="

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -n "benchmarks"
tmux send-keys -t "${SESSION}:benchmarks" "bash '${RUN_ALL}'" Enter

echo ""
echo "Attach : tmux attach -t ${SESSION}"
echo "Results: ${RESULTS_FILE}"
echo ""

if [[ -t 1 ]]; then
    tmux attach -t "${SESSION}"
fi
