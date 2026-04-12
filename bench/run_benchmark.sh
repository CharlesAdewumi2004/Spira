#!/usr/bin/env bash
# =============================================================================
# Spira benchmark runner (AWS-friendly, tmux, sequential-by-stage)
#
# Features:
#   - Detects repo root automatically
#   - Uses main repo for a branch if it is already checked out there
#   - Creates separate git worktrees for other stages
#   - Rebuilds each stage from a fresh build directory
#   - Runs benchmarks sequentially, one stage at a time
#   - Uses tmux with one window per stage
#   - Stage 1-3 pinned to one physical core
#   - Stage 4 pinned to all detected physical cores
#   - Writes one text log + one JSON file per stage
#
# Usage:
#   chmod +x bench/run_benchmarks_aws_tmux.sh
#   ./bench/run_benchmarks_aws_tmux.sh
#   ./bench/run_benchmarks_aws_tmux.sh --repetitions=10
#   ./bench/run_benchmarks_aws_tmux.sh --jobs=8
#   ./bench/run_benchmarks_aws_tmux.sh --target=spira_bench
#
# Notes:
#   - Designed for AWS / SSH use.
#   - Sequential execution is intentional for cleaner benchmark data.
#   - You can detach from tmux and reattach later.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 1. Resolve repo root robustly
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"

if [[ -z "${REPO_ROOT}" ]]; then
    echo "ERROR: Could not find git repository root from script location:" >&2
    echo "  ${SCRIPT_DIR}" >&2
    echo "Make sure this script is stored inside your Spira git repository." >&2
    exit 1
fi

if ! git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: ${REPO_ROOT} is not a valid git working tree." >&2
    exit 1
fi

WORKTREES_DIR="${REPO_ROOT}/.bench_worktrees"
RESULTS_DIR="${REPO_ROOT}/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.txt"

REPETITIONS=5
BUILD_JOBS="$(nproc)"
BENCH_TARGET="spira_bench"

# -----------------------------------------------------------------------------
# 2. Stage -> branch mapping
#    Edit these if your actual branch names differ.
# -----------------------------------------------------------------------------
declare -A STAGES=(
    [stage1]="stage1/MVP"
    [stage2]="stage2/SIMD-Kernels"
    [stage3]="stage3/layout-aware-csr"
    [stage4]="stage4/MutilThreaded"
)

STAGE_ORDER=(stage1 stage2 stage3 stage4)

# -----------------------------------------------------------------------------
# 3. Parse arguments
# -----------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --repetitions=*)
            REPETITIONS="${arg#*=}"
            ;;
        --jobs=*)
            BUILD_JOBS="${arg#*=}"
            ;;
        --target=*)
            BENCH_TARGET="${arg#*=}"
            ;;
        --help|-h)
            cat <<EOF
Usage:
  ./bench/run_benchmarks_aws_tmux.sh [options]

Options:
  --repetitions=N   Number of benchmark repetitions (default: 5)
  --jobs=N          Parallel build jobs for cmake --build (default: nproc)
  --target=NAME     Benchmark target/binary name (default: spira_bench)
  --help            Show this help
EOF
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

if ! [[ "${REPETITIONS}" =~ ^[0-9]+$ ]] || [[ "${REPETITIONS}" -lt 1 ]]; then
    echo "ERROR: --repetitions must be a positive integer." >&2
    exit 1
fi

if ! [[ "${BUILD_JOBS}" =~ ^[0-9]+$ ]] || [[ "${BUILD_JOBS}" -lt 1 ]]; then
    echo "ERROR: --jobs must be a positive integer." >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}" "${WORKTREES_DIR}"

# -----------------------------------------------------------------------------
# 4. Helpers
# -----------------------------------------------------------------------------
log_section() {
    local title="$1"
    {
        echo
        echo "=============================================================="
        echo "${title}"
        echo "=============================================================="
    } | tee -a "${RESULTS_FILE}"
}

run_cmd_logged() {
    local logfile="$1"
    shift

    "$@" > "${logfile}" 2>&1 || {
        cat "${logfile}" | tee -a "${RESULTS_FILE}"
        return 1
    }

    cat "${logfile}" >> "${RESULTS_FILE}"
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: ${cmd}" >&2
        exit 1
    fi
}

branch_in_use_at() {
    local branch="$1"
    git -C "${REPO_ROOT}" worktree list --porcelain | awk -v target="refs/heads/${branch}" '
        $1 == "worktree" { wt = $2 }
        $1 == "branch" && $2 == target { print wt; exit }
    '
}

# -----------------------------------------------------------------------------
# 5. Check required tools
# -----------------------------------------------------------------------------
require_cmd git
require_cmd cmake
require_cmd lscpu
require_cmd taskset
require_cmd tee
require_cmd awk
require_cmd tmux

# -----------------------------------------------------------------------------
# 6. Verify stage refs exist before doing work
# -----------------------------------------------------------------------------
for stage in "${STAGE_ORDER[@]}"; do
    branch="${STAGES[$stage]}"
    if ! git -C "${REPO_ROOT}" rev-parse --verify "${branch}" >/dev/null 2>&1; then
        echo "ERROR: Branch/ref does not exist for ${stage}: ${branch}" >&2
        echo "Check the STAGES mapping near the top of the script." >&2
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# 7. Detect CPU topology
# -----------------------------------------------------------------------------
echo "=== Detecting CPU topology ==="

declare -A PHYS_TO_CPU
while IFS=',' read -r cpu core; do
    [[ "${cpu}" =~ ^[0-9]+$ ]] || continue
    if [[ -n "${core}" && -z "${PHYS_TO_CPU[$core]+_}" ]]; then
        PHYS_TO_CPU["$core"]="${cpu}"
    fi
done < <(lscpu -p=cpu,core)

if [[ "${#PHYS_TO_CPU[@]}" -eq 0 ]]; then
    echo "ERROR: Failed to detect physical CPU topology from lscpu." >&2
    exit 1
fi

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

CPU_MODEL="$(lscpu | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
if [[ -z "${CPU_MODEL}" ]]; then
    CPU_MODEL="Unknown"
fi

# -----------------------------------------------------------------------------
# 8. Write metadata
# -----------------------------------------------------------------------------
{
    echo "Spira Benchmark Run — ${TIMESTAMP}"
    echo "Host                  : $(hostname)"
    echo "Kernel                : $(uname -r)"
    echo "CPU                   : ${CPU_MODEL}"
    echo "Architecture          : $(uname -m)"
    echo "Physical cores used   : ${NUM_PHYSICAL}"
    echo "Single-thread CPU     : ${SINGLE_CORE}"
    echo "Parallel CPU set      : ${PARALLEL_CPUS}"
    echo "Benchmark repetitions : ${REPETITIONS}"
    echo "Build jobs            : ${BUILD_JOBS}"
    echo "Benchmark target      : ${BENCH_TARGET}"
    echo "Repository root       : ${REPO_ROOT}"
    echo "Main repo commit      : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "CMake                 : $(cmake --version | head -n1)"
    echo "Compiler (c++)        : $(c++ --version 2>/dev/null | head -n1 || echo 'Unknown')"
    echo "=============================================================="
} > "${RESULTS_FILE}"

# -----------------------------------------------------------------------------
# 9. Resolve source dirs for each stage
# -----------------------------------------------------------------------------
declare -A STAGE_SRCDIRS
declare -A STAGE_COMMITS
declare -A BENCH_BINS
declare -A BUILD_DIRS

log_section "Preparing stage source trees"

for stage in "${STAGE_ORDER[@]}"; do
    branch="${STAGES[$stage]}"
    desired_wt_dir="${WORKTREES_DIR}/${stage}"

    log_section "Prepare: ${stage} (${branch})"

    in_use_path="$(branch_in_use_at "${branch}" || true)"

    if [[ -n "${in_use_path}" ]]; then
        echo "Branch ${branch} is already checked out at: ${in_use_path}" | tee -a "${RESULTS_FILE}"
        src_dir="${in_use_path}"
    else
        src_dir="${desired_wt_dir}"
        if [[ ! -d "${src_dir}" ]]; then
            git -C "${REPO_ROOT}" worktree add "${src_dir}" "${branch}" | tee -a "${RESULTS_FILE}"
        else
            echo "Worktree already exists: ${src_dir}" | tee -a "${RESULTS_FILE}"
            git -C "${src_dir}" checkout "${branch}" >> "${RESULTS_FILE}" 2>&1 || {
                echo "ERROR: Failed to checkout branch '${branch}' in ${src_dir}" | tee -a "${RESULTS_FILE}" >&2
                exit 1
            }
        fi
    fi

    STAGE_SRCDIRS["${stage}"]="${src_dir}"
    STAGE_COMMITS["${stage}"]="$(git -C "${src_dir}" rev-parse HEAD)"

    if [[ "${src_dir}" == "${REPO_ROOT}" ]]; then
        build_dir="${WORKTREES_DIR}/builds/${stage}"
    else
        build_dir="${src_dir}/build_bench"
    fi

    BUILD_DIRS["${stage}"]="${build_dir}"

    echo "Source dir  : ${src_dir}" | tee -a "${RESULTS_FILE}"
    echo "Build dir   : ${build_dir}" | tee -a "${RESULTS_FILE}"
    echo "Stage commit: ${STAGE_COMMITS[$stage]}" | tee -a "${RESULTS_FILE}"
done

# -----------------------------------------------------------------------------
# 10. Build each stage
# -----------------------------------------------------------------------------
log_section "Building stages"

for stage in "${STAGE_ORDER[@]}"; do
    src_dir="${STAGE_SRCDIRS[$stage]}"
    build_dir="${BUILD_DIRS[$stage]}"
    bin="${build_dir}/${BENCH_TARGET}"

    log_section "Build: ${stage}"
    {
        echo "Branch    : ${STAGES[$stage]}"
        echo "Commit    : ${STAGE_COMMITS[$stage]}"
        echo "Source dir: ${src_dir}"
        echo "Build dir : ${build_dir}"
    } | tee -a "${RESULTS_FILE}"

    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"

    configure_log="${build_dir}/cmake_configure.log"
    build_log="${build_dir}/cmake_build.log"

    echo "Configuring ${stage}..." | tee -a "${RESULTS_FILE}"
    run_cmd_logged "${configure_log}" \
        cmake -S "${src_dir}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPIRA_BUILD_TESTS=OFF \
        -DSPIRA_BUILD_BENCHMARKS=ON \
        -DSPIRA_ENABLE_SIMD=ON

    echo "Building ${stage}..." | tee -a "${RESULTS_FILE}"
    run_cmd_logged "${build_log}" \
        cmake --build "${build_dir}" --target "${BENCH_TARGET}" -j"${BUILD_JOBS}"

    if [[ ! -x "${bin}" ]]; then
        echo "ERROR: Benchmark binary not found for ${stage}: ${bin}" | tee -a "${RESULTS_FILE}" >&2
        echo "If the binary name differs from '${BENCH_TARGET}', rerun with:" | tee -a "${RESULTS_FILE}" >&2
        echo "  --target=<actual_binary_name>" | tee -a "${RESULTS_FILE}" >&2
        exit 1
    fi

    BENCH_BINS["${stage}"]="${bin}"
    echo "Built binary: ${bin}" | tee -a "${RESULTS_FILE}"
done

# -----------------------------------------------------------------------------
# 11. Generate per-stage runner scripts
# -----------------------------------------------------------------------------
RUNNER_DIR="${WORKTREES_DIR}/runners_${TIMESTAMP}"
mkdir -p "${RUNNER_DIR}"

log_section "Generating tmux runner scripts"

for idx in "${!STAGE_ORDER[@]}"; do
    stage="${STAGE_ORDER[$idx]}"
    bin="${BENCH_BINS[$stage]}"
    json_file="${RESULTS_FILE%.txt}_${stage}.json"

    if [[ "${stage}" == "stage4" ]]; then
        cpus="${PARALLEL_CPUS}"
        pin_desc="CPUs ${PARALLEL_CPUS} (${NUM_PHYSICAL} physical cores)"
    else
        cpus="${SINGLE_CORE}"
        pin_desc="CPU ${SINGLE_CORE} (single physical core)"
    fi

    prev_signal=""
    if [[ "${idx}" -gt 0 ]]; then
        prev_stage="${STAGE_ORDER[$((idx - 1))]}"
        prev_signal="${prev_stage}_done_${TIMESTAMP}"
    fi
    done_signal="${stage}_done_${TIMESTAMP}"

    runner="${RUNNER_DIR}/${stage}.sh"

    cat > "${runner}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

RESULTS_FILE='${RESULTS_FILE}'
STAGE='${stage}'
BRANCH='${STAGES[$stage]}'
COMMIT='${STAGE_COMMITS[$stage]}'
SOURCE_DIR='${STAGE_SRCDIRS[$stage]}'
BIN='${bin}'
JSON_FILE='${json_file}'
CPU_DESC='${pin_desc}'
CPUS='${cpus}'
PREV_SIGNAL='${prev_signal}'
DONE_SIGNAL='${done_signal}'
REPETITIONS='${REPETITIONS}'

if [[ -n "\${PREV_SIGNAL}" ]]; then
    echo "[\${STAGE}] Waiting for previous stage signal: \${PREV_SIGNAL}"
    tmux wait-for "\${PREV_SIGNAL}"
fi

{
    echo
    echo "=============================================================="
    echo "Run: \${STAGE}"
    echo "=============================================================="
    echo "Branch   : \${BRANCH}"
    echo "Commit   : \${COMMIT}"
    echo "Source   : \${SOURCE_DIR}"
    echo "Binary   : \${BIN}"
    echo "Pinned   : \${CPU_DESC}"
    echo "JSON out : \${JSON_FILE}"
    echo "Start    : \$(date)"
    echo
} | tee -a "\${RESULTS_FILE}"

taskset -c "\${CPUS}" "\${BIN}" \\
    --benchmark_format=console \\
    --benchmark_repetitions="\${REPETITIONS}" \\
    --benchmark_report_aggregates_only=true \\
    --benchmark_out="\${JSON_FILE}" \\
    --benchmark_out_format=json \\
    2>&1 | tee -a "\${RESULTS_FILE}"

{
    echo
    echo "End      : \$(date)"
    echo "Finished : \${STAGE}"
} | tee -a "\${RESULTS_FILE}"

tmux wait-for -S "\${DONE_SIGNAL}"

echo
echo "[\${STAGE}] Done. Signal sent: \${DONE_SIGNAL}"
echo "[\${STAGE}] Press Ctrl+b then d to detach, or just leave this window open."
EOF

    chmod +x "${runner}"
    echo "Runner created: ${runner}" | tee -a "${RESULTS_FILE}"
done

# -----------------------------------------------------------------------------
# 12. Launch tmux session
# -----------------------------------------------------------------------------
SESSION="spira_bench_${TIMESTAMP}"

log_section "Launching tmux session"

# Clean up any unlikely collision
tmux kill-session -t "${SESSION}" 2>/dev/null || true

# First window / session
first_stage="${STAGE_ORDER[0]}"
tmux new-session -d -s "${SESSION}" -n "${first_stage}"
tmux send-keys -t "${SESSION}:${first_stage}" "bash '${RUNNER_DIR}/${first_stage}.sh'" C-m

# Remaining windows
for idx in "${!STAGE_ORDER[@]}"; do
    if [[ "${idx}" -eq 0 ]]; then
        continue
    fi
    stage="${STAGE_ORDER[$idx]}"
    tmux new-window -t "${SESSION}" -n "${stage}"
    tmux send-keys -t "${SESSION}:${stage}" "bash '${RUNNER_DIR}/${stage}.sh'" C-m
done

tmux select-window -t "${SESSION}:${first_stage}"

# -----------------------------------------------------------------------------
# 13. Final summary
# -----------------------------------------------------------------------------
log_section "Started"

echo "tmux session started:"
echo "  ${SESSION}"
echo
echo "Attach with:"
echo "  tmux attach -t ${SESSION}"
echo
echo "Detach later with:"
echo "  Ctrl+b then d"
echo
echo "Results file:"
echo "  ${RESULTS_FILE}"
echo
echo "JSON files:"
for stage in "${STAGE_ORDER[@]}"; do
    echo "  ${RESULTS_FILE%.txt}_${stage}.json"
done
echo
echo "The stages will run strictly in this order:"
echo "  ${STAGE_ORDER[*]}"
echo
echo "Copy results back later with:"
echo "  scp -i <key.pem> ubuntu@<server-ip>:${RESULTS_FILE} ."
echo "  scp -i <key.pem> ubuntu@<server-ip>:${RESULTS_FILE%.txt}_stage*.json ."