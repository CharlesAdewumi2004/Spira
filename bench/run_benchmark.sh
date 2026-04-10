#!/usr/bin/env bash
# =============================================================================
# Spira benchmark runner — AWS r8i.4xlarge (or similar)
#
# Builds all 4 stages in git worktrees, then launches a tmux session with one
# window per stage.  Single-threaded stages (1-3) are pinned to one physical
# core; stage 4 (parallel) gets all physical cores with HT disabled.
#
# Usage:
#   chmod +x bench/run_benchmarks.sh
#   ./bench/run_benchmarks.sh [--no-smt-disable]
#
# Results are saved to: bench_results/results_<timestamp>.txt
# Retrieve with:  scp -i <key> ubuntu@<ip>:~/spira/bench_results/*.txt .
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.txt"
WORKTREES_DIR="${REPO_ROOT}/.bench_worktrees"
SESSION="spira_bench"

# Branch → directory name mapping
declare -A STAGES=(
    [stage1]="stage1/MVP"
    [stage2]="stage2/SIMD-Kernels"
    [stage3]="stage3/layout-aware-csr"
    [stage4]="stage4/MutilThreaded"
)
# Ordered for tmux windows
STAGE_ORDER=(stage1 stage2 stage3 stage4)

DISABLE_SMT=true
for arg in "$@"; do
    [[ "$arg" == "--no-smt-disable" ]] && DISABLE_SMT=false
done

# =============================================================================
# 1. Detect CPU topology
# =============================================================================
echo "=== Detecting CPU topology ==="

# Parse lscpu -p=cpu,core to build: physical_core_id -> first_cpu mapping.
# Each physical core may have 2+ logical CPUs (HT siblings); we keep only
# the first logical CPU we see for each physical core.
declare -A PHYS_TO_CPU
while IFS=',' read -r cpu core; do
    [[ "$cpu" =~ ^[0-9]+$ ]] || continue   # skip header lines
    if [[ -z "${PHYS_TO_CPU[$core]+_}" ]]; then
        PHYS_TO_CPU["$core"]="$cpu"
    fi
done < <(lscpu -p=cpu,core)

PHYSICAL_CPUS=()
for core in $(printf '%s\n' "${!PHYS_TO_CPU[@]}" | sort -n); do
    PHYSICAL_CPUS+=("${PHYS_TO_CPU[$core]}")
done

NUM_PHYSICAL="${#PHYSICAL_CPUS[@]}"
SINGLE_CORE="${PHYSICAL_CPUS[0]}"
PARALLEL_CPUS="$(IFS=','; echo "${PHYSICAL_CPUS[*]}")"

echo "  Physical cores found : ${NUM_PHYSICAL}"
echo "  Single-thread pin    : CPU ${SINGLE_CORE}"
echo "  Parallel CPUs        : ${PARALLEL_CPUS}"

# =============================================================================
# 2. Optionally disable SMT (requires root)
# =============================================================================
SMT_WAS_ON=false
SMT_CTL="/sys/devices/system/cpu/smt/control"

if [[ "$DISABLE_SMT" == true ]]; then
    if [[ -f "$SMT_CTL" ]]; then
        if [[ "$(cat "$SMT_CTL")" != "off" ]]; then
            if [[ $EUID -eq 0 ]]; then
                echo "=== Disabling SMT/HT ==="
                echo off > "$SMT_CTL"
                SMT_WAS_ON=true
                echo "  SMT disabled."
            else
                echo "WARNING: Not root — cannot disable SMT. Running with HT enabled."
                echo "         Re-run with sudo or pass --no-smt-disable to suppress this warning."
            fi
        else
            echo "=== SMT already off ==="
        fi
    else
        echo "=== SMT control not available (non-x86 or kernel too old) ==="
    fi
fi

# Re-read CPU list after possible SMT change (online CPUs may have changed)
if [[ "$SMT_WAS_ON" == true ]]; then
    unset PHYS_TO_CPU
    declare -A PHYS_TO_CPU
    while IFS=',' read -r cpu core; do
        [[ "$cpu" =~ ^[0-9]+$ ]] || continue
        if [[ -z "${PHYS_TO_CPU[$core]+_}" ]]; then
            PHYS_TO_CPU["$core"]="$cpu"
        fi
    done < <(lscpu -p=cpu,core)
    PHYSICAL_CPUS=()
    for core in $(printf '%s\n' "${!PHYS_TO_CPU[@]}" | sort -n); do
        PHYSICAL_CPUS+=("${PHYS_TO_CPU[$core]}")
    done
    NUM_PHYSICAL="${#PHYSICAL_CPUS[@]}"
    SINGLE_CORE="${PHYSICAL_CPUS[0]}"
    PARALLEL_CPUS="$(IFS=','; echo "${PHYSICAL_CPUS[*]}")"
    echo "  After SMT disable — Parallel CPUs: ${PARALLEL_CPUS}"
fi

# =============================================================================
# 3. Create results dir
# =============================================================================
mkdir -p "${RESULTS_DIR}"
echo "Results → ${RESULTS_FILE}"

# =============================================================================
# 4. Build each stage in its own worktree
# =============================================================================
echo ""
echo "=== Building stages ==="

declare -A BENCH_BINS

for stage in "${STAGE_ORDER[@]}"; do
    branch="${STAGES[$stage]}"
    wt_dir="${WORKTREES_DIR}/${stage}"
    build_dir="${wt_dir}/build_bench"
    bin="${build_dir}/spira_bench"

    echo ""
    echo "--- ${stage}: ${branch} ---"

    # Add worktree if it doesn't exist yet
    if [[ ! -d "$wt_dir" ]]; then
        git -C "${REPO_ROOT}" worktree add "${wt_dir}" "${branch}"
    else
        echo "  Worktree exists, updating..."
        git -C "${wt_dir}" checkout "${branch}" 2>/dev/null || true
        git -C "${wt_dir}" pull --ff-only 2>/dev/null || true
    fi

    mkdir -p "${build_dir}"

    cmake -S "${wt_dir}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPIRA_BUILD_TESTS=OFF \
        -DSPIRA_BUILD_BENCHMARKS=ON \
        -DSPIRA_ENABLE_SIMD=ON \
        2>&1 | tail -5

    cmake --build "${build_dir}" --target spira_bench -j"$(nproc)" 2>&1 | tail -10

    if [[ ! -f "$bin" ]]; then
        echo "ERROR: Build failed for ${stage} — binary not found at ${bin}" >&2
        exit 1
    fi

    BENCH_BINS["$stage"]="$bin"
    echo "  Built: ${bin}"
done

# =============================================================================
# 5. Write per-stage runner scripts (sourced inside tmux windows)
# =============================================================================
RUNNER_DIR="${WORKTREES_DIR}/runners"
mkdir -p "${RUNNER_DIR}"

for stage in "${STAGE_ORDER[@]}"; do
    bin="${BENCH_BINS[$stage]}"
    runner="${RUNNER_DIR}/${stage}.sh"

    if [[ "$stage" == "stage4" ]]; then
        cpus="${PARALLEL_CPUS}"
        cpu_label="CPUs ${PARALLEL_CPUS} (${NUM_PHYSICAL} physical cores)"
    else
        cpus="${SINGLE_CORE}"
        cpu_label="CPU ${SINGLE_CORE} (single physical core)"
    fi

    cat > "${runner}" <<-RUNNER
		#!/usr/bin/env bash
		set -euo pipefail
		STAGE="${stage}"
		BIN="${bin}"
		CPU_LABEL="${cpu_label}"
		CPUS="${cpus}"
		RESULTS_FILE="${RESULTS_FILE}"

		echo ""
		echo "============================================================"
		echo "  \$STAGE — \$(date)"
		echo "  Binary : \$BIN"
		echo "  Pinned : \$CPU_LABEL"
		echo "============================================================"

		# Header into results file
		{
		    echo ""
		    echo "============================================================"
		    echo "  \$STAGE — \$(date)"
		    echo "  Binary : \$BIN"
		    echo "  Pinned : \$CPU_LABEL"
		    echo "============================================================"
		} >> "\$RESULTS_FILE"

		# Run benchmark — tee to console and results file
		taskset -c "\$CPUS" "\$BIN" \\
		    --benchmark_format=console \\
		    --benchmark_repetitions=5 \\
		    --benchmark_report_aggregates_only=true \\
		    --benchmark_out="\${RESULTS_FILE%.txt}_\${STAGE}.json" \\
		    --benchmark_out_format=json \\
		    2>&1 | tee -a "\$RESULTS_FILE"

		echo ""
		echo "\$STAGE DONE. Results appended to \$RESULTS_FILE"
		echo "JSON saved to \${RESULTS_FILE%.txt}_\${STAGE}.json"
		echo ""
		echo "Press any key to close this window..."
		read -r -n1
		RUNNER
    chmod +x "${runner}"
done

# =============================================================================
# 6. Launch tmux session
# =============================================================================
echo ""
echo "=== Launching tmux session '${SESSION}' ==="

# Kill existing session if present
tmux kill-session -t "${SESSION}" 2>/dev/null || true

# Write a summary header to the results file
{
    echo "Spira Benchmark Run — ${TIMESTAMP}"
    echo "Host     : $(hostname)"
    echo "Kernel   : $(uname -r)"
    echo "CPU      : $(lscpu | grep 'Model name' | sed 's/Model name:\s*//')"
    echo "Physical cores used (parallel): ${NUM_PHYSICAL}"
    echo "Single-thread pin: CPU ${SINGLE_CORE}"
    echo "SMT disabled: ${SMT_WAS_ON}"
    echo "--------------------------------------------------------------"
} > "${RESULTS_FILE}"

# Create session with first window
tmux new-session -d -s "${SESSION}" -n "${STAGE_ORDER[0]}"
tmux send-keys -t "${SESSION}:${STAGE_ORDER[0]}" \
    "bash '${RUNNER_DIR}/${STAGE_ORDER[0]}.sh'" Enter

# Add remaining windows
for i in 1 2 3; do
    stage="${STAGE_ORDER[$i]}"
    tmux new-window -t "${SESSION}" -n "${stage}"
    tmux send-keys -t "${SESSION}:${stage}" \
        "bash '${RUNNER_DIR}/${stage}.sh'" Enter
done

# Switch back to stage1 window
tmux select-window -t "${SESSION}:${STAGE_ORDER[0]}"

echo ""
echo "tmux session '${SESSION}' started."
echo ""
echo "Attach with:   tmux attach -t ${SESSION}"
echo "Results file:  ${RESULTS_FILE}"
echo "JSON files:    ${RESULTS_FILE%.txt}_stage{1,2,3,4}.json"
echo ""
echo "When done, retrieve results with:"
echo "  scp -i <key> ubuntu@<ip>:${RESULTS_FILE} ."
echo "  scp -i <key> ubuntu@<ip>:${RESULTS_FILE%.txt}_stage*.json ."
echo ""

# Optionally attach immediately if we're in an interactive terminal
if [[ -t 1 && "${AUTO_ATTACH:-true}" == "true" ]]; then
    tmux attach -t "${SESSION}"
fi

# =============================================================================
# 7. Restore SMT on exit (trap)
# =============================================================================
restore_smt() {
    if [[ "$SMT_WAS_ON" == true && -f "$SMT_CTL" && $EUID -eq 0 ]]; then
        echo "Restoring SMT..."
        echo on > "$SMT_CTL"
    fi
}
trap restore_smt EXIT
