#!/usr/bin/env bash
set -euo pipefail

echo ">>> EXITING BENCHMARK MODE (restoring defaults)"

if [[ "$EUID" -ne 0 ]]; then
  echo "!!! This script must be run as root (sudo)."
  exit 1
fi

# Restore governor to ondemand or powersave (pick what you prefer)
if ls /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor >/dev/null 2>&1; then
  echo " - Setting CPU scaling governor back to 'ondemand' (or change to 'schedutil' if you prefer)"
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo ondemand > "$f" 2>/dev/null || echo schedutil > "$f" 2>/dev/null || true
  done
fi

# Re-enable turbo / boost where possible
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
  echo " - Re-enabling Intel Turbo Boost"
  echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo || true
fi

if [ -f /sys/devices/system/cpu/intel_pstate/hwp_enabled ]; then
  echo " - Re-enabling Intel HWP"
  echo 1 > /sys/devices/system/cpu/intel_pstate/hwp_enabled || true
fi

if [ -f /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost ]; then
  echo " - Re-enabling Intel HWP dynamic boost"
  echo 1 > /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost || true
fi

if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
  echo " - Re-enabling generic CPU boost"
  echo 1 > /sys/devices/system/cpu/cpufreq/boost || true
fi

# Re-enable C-states
if ls /sys/devices/system/cpu/cpu0/cpuidle/state*/disable >/dev/null 2>&1; then
  echo " - Re-enabling CPU idle states"
  for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 0 > "$s" || true
  done
fi

# Re-enable ASLR
if command -v sysctl >/dev/null 2>&1; then
  echo " - Re-enabling ASLR (kernel.randomize_va_space=2)"
  sysctl -w kernel.randomize_va_space=2 >/dev/null || true
fi

# Re-enable SMT (if we turned it off)
if [ -f /sys/devices/system/cpu/smt/control ]; then
  echo " - Re-enabling SMT (Hyper-Threading)"
  echo on > /sys/devices/system/cpu/smt/control || true
fi

echo ">>> System restored to normal mode (reboot also works if in doubt)."
