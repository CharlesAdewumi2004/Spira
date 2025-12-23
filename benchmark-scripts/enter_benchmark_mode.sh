#!/usr/bin/env bash
set -euo pipefail

echo ">>> ENTERING BENCHMARK MODE"

if [[ "$EUID" -ne 0 ]]; then
  echo "!!! This script must be run as root (sudo)."
  exit 1
fi

# -------------------------------------------
# 0. Basic CPU info (just for sanity)
# -------------------------------------------
if command -v lscpu >/dev/null 2>&1; then
  echo "CPU info:"
  lscpu | egrep 'Model name|CPU\(s\)|Thread|Core|Socket|MHz' || true
fi
echo

# -------------------------------------------
# 1. Set governor to performance
# -------------------------------------------
if ls /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor >/dev/null 2>&1; then
  echo " - Setting CPU scaling governor to 'performance'"
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$f" || true
  done
else
  echo " ! No cpufreq governor interface found; skipping governor change"
fi

# -------------------------------------------
# 2. Disable Turbo Boost / Boost
# -------------------------------------------
# Intel pstate turbo
if [ -d /sys/devices/system/cpu/intel_pstate ]; then
  if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo " - Disabling Intel Turbo Boost"
    echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo || true
  fi

  # HWP / Speed Shift knobs
  if [ -f /sys/devices/system/cpu/intel_pstate/hwp_enabled ]; then
    echo " - Disabling Intel HWP (Speed Shift)"
    echo 0 > /sys/devices/system/cpu/intel_pstate/hwp_enabled || true
  fi

  if [ -f /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost ]; then
    echo " - Disabling Intel HWP dynamic boost"
    echo 0 > /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost || true
  fi
fi

# AMD / generic boost (may or may not exist)
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
  echo " - Disabling generic CPU boost"
  echo 0 > /sys/devices/system/cpu/cpufreq/boost || true
fi

# -------------------------------------------
# 3. Lock CPU frequency to a fixed value
# -------------------------------------------
# Pick something sensible for your chip; you can tweak this.
FREQ_KHZ=2600000   # 2.6 GHz

if ls /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq >/dev/null 2>&1; then
  echo " - Locking CPU frequency at ${FREQ_KHZ} kHz (min/max)"
  for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -d "$cpu/cpufreq" ]; then
      echo "$FREQ_KHZ" > "$cpu/cpufreq/scaling_min_freq" || true
      echo "$FREQ_KHZ" > "$cpu/cpufreq/scaling_max_freq" || true
    fi
  done
else
  echo " ! cpufreq min/max not available; cannot hard-lock frequency"
fi

# -------------------------------------------
# 4. Disable deep C-states (idle states)
# -------------------------------------------
if ls /sys/devices/system/cpu/cpu0/cpuidle/state*/disable >/dev/null 2>&1; then
  echo " - Disabling CPU idle states (C-states > C0)"
  for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 > "$s" || true
  done
else
  echo " ! No cpuidle states interface found; skipping C-state disable"
fi

# -------------------------------------------
# 5. Disable ASLR (for more repeatable memory layout)
# -------------------------------------------
if command -v sysctl >/dev/null 2>&1; then
  echo " - Disabling ASLR (kernel.randomize_va_space=0)"
  sysctl -w kernel.randomize_va_space=0 >/dev/null || true
else
  echo " ! sysctl not available; cannot change ASLR"
fi

# -------------------------------------------
# 6. Optionally disable SMT (Hyper-Threading)
#    (You can comment this out if you want SMT on)
# -------------------------------------------
if [ -f /sys/devices/system/cpu/smt/control ]; then
  echo " - Disabling SMT (Hyper-Threading)"
  echo off > /sys/devices/system/cpu/smt/control || true
else
  echo " ! No SMT control interface; skipping SMT disable"
fi

echo
echo ">>> Benchmark mode enabled."
echo
echo "Suggested usage:"
echo "  # On HOST: run container pinned to one core, e.g.:"
echo "    docker run --rm --cpuset-cpus=\"0\" -it your-image-name"
echo
echo "  # Inside container: run benchmarks pinned to that core:"
echo "    taskset -c 0 ./build/BM_spmv \\"
echo "      --benchmark_min_time=2.0 \\"
echo "      --benchmark_repetitions=20 \\"
echo "      --benchmark_report_aggregates_only=true"
echo
echo "When finished, run ./exit_benchmark_mode.sh or reboot to restore settings."
