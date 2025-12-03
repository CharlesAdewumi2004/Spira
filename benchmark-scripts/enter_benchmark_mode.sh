#!/usr/bin/env bash
set -euo pipefail

echo ">>> ENTERING BENCHMARK MODE (host)"

# -------------------------------
# 1. Disable Turbo Boost (Intel)
# -------------------------------
if [ -d /sys/devices/system/cpu/intel_pstate ]; then
  if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo " - Disabling turbo boost"
    echo 1 | tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
  fi

  # -------------------------------------------
  # 2. Disable Intel Speed Shift (HWP) if any
  # -------------------------------------------
  if [ -f /sys/devices/system/cpu/intel_pstate/hwp_enabled ]; then
    echo " - Disabling HWP (Speed Shift)"
    echo 0 | tee /sys/devices/system/cpu/intel_pstate/hwp_enabled >/dev/null
  fi

  if [ -f /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost ]; then
    echo " - Disabling HWP dynamic boost"
    echo 0 | tee /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost >/dev/null
  fi
fi

# -------------------------------------------
# 3. Lock CPU frequency (pick a sensible freq)
# -------------------------------------------
# You can change this to 2600000, 3000000, etc.
FREQ=2600000
if ls /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq >/dev/null 2>&1; then
  echo " - Locking CPU frequency at ${FREQ} kHz"
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    echo "$FREQ" | tee "$f" >/dev/null
  done
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
    echo "$FREQ" | tee "$f" >/dev/null
  done
else
  echo " ! cpufreq not available, skipping fixed frequency lock"
fi

# -------------------------------------------
# 4. Disable CPU C-states (stop deep sleep)
# -------------------------------------------
if ls /sys/devices/system/cpu/cpu0/cpuidle/state*/disable >/dev/null 2>&1; then
  echo " - Disabling CPU idle states (C-states)"
  for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 | tee "$s" >/dev/null || true
  done
else
  echo " ! No cpuidle states found, skipping C-state disable"
fi

# -------------------------------------------
# 5. Disable ASLR (more repeatable memory layout)
# -------------------------------------------
echo " - Disabling ASLR"
sysctl -w kernel.randomize_va_space=0 >/dev/null

echo ">>> Benchmark mode enabled."
echo ">>> When running benchmarks, use e.g.:"
echo "    taskset -c 0 ./build/BM_spmv --benchmark_repetitions=20 --benchmark_report_aggregates_only=true"
echo ">>> When finished, run ./exit_benchmark_mode.sh OR reboot."
