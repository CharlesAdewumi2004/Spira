#!/usr/bin/env bash
set -euo pipefail

echo ">>> EXITING BENCHMARK MODE (host)"

# -------------------------------
# 1. Re-enable Turbo Boost
# -------------------------------
if [ -d /sys/devices/system/cpu/intel_pstate ]; then
  if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo " - Enabling turbo boost"
    echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
  fi

  # -------------------------------------------
  # 2. Re-enable Intel Speed Shift (HWP)
  # -------------------------------------------
  if [ -f /sys/devices/system/cpu/intel_pstate/hwp_enabled ]; then
    echo " - Enabling HWP (Speed Shift)"
    echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/hwp_enabled >/dev/null
  fi

  if [ -f /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost ]; then
    echo " - Enabling HWP dynamic boost"
    echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost >/dev/null
  fi
fi

# -------------------------------------------
# 3. Let cpufreq go back to auto control
# -------------------------------------------
if ls /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq >/dev/null 2>&1; then
  echo " - Resetting scaling_min/max_freq to 0 (driver default)"
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    echo 0 | sudo tee "$f" >/dev/null || true
  done
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
    echo 0 | sudo tee "$f" >/dev/null || true
  done
fi

# -------------------------------------------
# 4. Re-enable C-states
# -------------------------------------------
if ls /sys/devices/system/cpu/cpu0/cpuidle/state*/disable >/dev/null 2>&1; then
  echo " - Re-enabling CPU idle states (C-states)"
  for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 0 | sudo tee "$s" >/dev/null || true
  done
fi

# -------------------------------------------
# 5. Re-enable ASLR
# -------------------------------------------
echo " - Enabling ASLR"
sudo sysctl -w kernel.randomize_va_space=2 >/dev/null

echo ">>> Benchmark mode disabled."
echo ">>> For a completely clean slate, a reboot is still recommended."
