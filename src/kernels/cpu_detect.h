#ifndef SPIRA_KERNELS_CPU_DETECT_H
#define SPIRA_KERNELS_CPU_DETECT_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// Platform detection
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #ifndef SPIRA_ARCH_X86
    #define SPIRA_ARCH_X86 1
  #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  #ifndef SPIRA_ARCH_ARM64
    #define SPIRA_ARCH_ARM64 1
  #endif
#elif defined(__arm__) || defined(_M_ARM)
  #ifndef SPIRA_ARCH_ARM32
    #define SPIRA_ARCH_ARM32 1
  #endif
#endif

// ============================================================================
// CPUID includes
// ============================================================================

#if defined(SPIRA_ARCH_X86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_MSC_VER)
#include <processthreadsapi.h>
#include <sysinfoapi.h>
#endif
#endif

namespace spira::kernel {

// ============================================================================
// Cache size information
// ============================================================================

struct CacheInfo {
    uint32_t l1d_size  = 0;  // L1 data cache in bytes        (0 = unknown)
    uint32_t l1i_size  = 0;  // L1 instruction cache in bytes (0 = unknown)
    uint32_t l2_size   = 0;  // L2 unified cache in bytes     (0 = unknown)
    uint32_t l3_size   = 0;  // L3 unified cache in bytes     (0 = unknown)
    uint32_t line_size = 0;  // Cache line size in bytes      (0 = unknown)
};

struct CpuFeatures {
    // x86 features
    bool sse2 = false;
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;  // foundation
    bool avx512bw = false; // byte/word
    bool avx512vl = false; // vector length extensions
    bool avx512dq = false; // doubleword/quadword

    // ARM features
    bool neon = false;
    bool sve = false;
    bool sve2 = false;

    // Cache topology (populated for all architectures)
    CacheInfo cache;

    CpuFeatures() { detect(); }

  private:
    void detect() {
#if defined(SPIRA_ARCH_X86)
        detect_x86();
#elif defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)
        detect_arm();
#endif
        detect_cache();
    }

    void detect_cache() {
#if defined(SPIRA_ARCH_X86)
        detect_cache_x86();
#elif defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)
        detect_cache_arm();
#endif
    }

#if defined(SPIRA_ARCH_X86)

    // ========================================================================
    // x86 detection
    // ========================================================================

    struct CpuidResult {
        uint32_t eax, ebx, ecx, edx;
    };

    static CpuidResult cpuid(uint32_t leaf, uint32_t subleaf = 0) {
        CpuidResult r{};
#if defined(_MSC_VER)
        int regs[4];
        __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
        r.eax = static_cast<uint32_t>(regs[0]);
        r.ebx = static_cast<uint32_t>(regs[1]);
        r.ecx = static_cast<uint32_t>(regs[2]);
        r.edx = static_cast<uint32_t>(regs[3]);
#else
        __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
        return r;
    }

    static uint64_t xgetbv(uint32_t index) {
#if defined(_MSC_VER)
        return _xgetbv(index);
#else
        uint32_t eax, edx;
        __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
        return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
    }

    void detect_x86() {
        // Check max supported CPUID leaf
        auto leaf0 = cpuid(0);
        uint32_t max_leaf = leaf0.eax;

        if (max_leaf < 1)
            return;

        // Leaf 1: basic features
        auto leaf1 = cpuid(1);
        sse2 = (leaf1.edx >> 26) & 1;          // EDX bit 26
        sse42 = (leaf1.ecx >> 20) & 1;         // ECX bit 20
        bool os_xsave = (leaf1.ecx >> 27) & 1; // ECX bit 27 — OS supports XSAVE
        avx = (leaf1.ecx >> 28) & 1;           // ECX bit 28
        fma = (leaf1.ecx >> 12) & 1;           // ECX bit 12

        // AVX/FMA require OS support for saving YMM registers
        if (avx && os_xsave) {
            uint64_t xcr0 = xgetbv(0);
            bool os_saves_xmm = (xcr0 & 0x2) != 0; // bit 1: XMM state
            bool os_saves_ymm = (xcr0 & 0x4) != 0; // bit 2: YMM state

            if (!os_saves_xmm || !os_saves_ymm) {
                avx = false;
                fma = false;
            }
        } else {
            // No OS XSAVE support — disable AVX and everything above
            avx = false;
            fma = false;
        }

        // Leaf 7: extended features (requires AVX to be usable)
        if (max_leaf >= 7 && avx) {
            auto leaf7 = cpuid(7, 0);
            avx2 = (leaf7.ebx >> 5) & 1;      // EBX bit 5
            avx512f = (leaf7.ebx >> 16) & 1;  // EBX bit 16
            avx512bw = (leaf7.ebx >> 30) & 1; // EBX bit 30
            avx512vl = (leaf7.ebx >> 31) & 1; // EBX bit 31
            avx512dq = (leaf7.ebx >> 17) & 1; // EBX bit 17

            // AVX-512 requires OS support for ZMM registers and opmask
            if (avx512f) {
                uint64_t xcr0 = xgetbv(0);
                bool os_saves_opmask = (xcr0 & 0x20) != 0; // bit 5
                bool os_saves_zmm_lo = (xcr0 & 0x40) != 0; // bit 6
                bool os_saves_zmm_hi = (xcr0 & 0x80) != 0; // bit 7

                if (!os_saves_opmask || !os_saves_zmm_lo || !os_saves_zmm_hi) {
                    avx512f = false;
                    avx512bw = false;
                    avx512vl = false;
                    avx512dq = false;
                }
            }
        } else {
            avx2 = false;
            avx512f = avx512bw = avx512vl = avx512dq = false;
        }
    }

    void detect_cache_x86() {
        // CPUID leaf 4: deterministic cache parameters — enumerate subleaves
        // until cache type field (bits 4:0 of EAX) is zero (null).
        auto leaf0 = cpuid(0);
        if (leaf0.eax < 4) return;

        for (uint32_t sub = 0; sub < 32; ++sub) {
            auto r = cpuid(4, sub);
            uint32_t type = r.eax & 0x1F;  // 0=null, 1=data, 2=instruction, 3=unified
            if (type == 0) break;

            uint32_t level      = (r.eax >> 5) & 0x7;          // bits 7:5
            uint32_t line_size  = (r.ebx & 0xFFF) + 1;         // bits 11:0
            uint32_t partitions = ((r.ebx >> 12) & 0x3FF) + 1; // bits 21:12
            uint32_t ways       = ((r.ebx >> 22) & 0x3FF) + 1; // bits 31:22
            uint32_t sets       = r.ecx + 1;

            uint32_t size = line_size * partitions * ways * sets;

            if (cache.line_size == 0) cache.line_size = line_size;

            if (level == 1) {
                if (type == 1 || type == 3) cache.l1d_size = size; // data or unified
                if (type == 2 || type == 3) cache.l1i_size = size; // instr or unified
            } else if (level == 2) {
                cache.l2_size = size;
            } else if (level == 3) {
                cache.l3_size = size;
            }
        }
    }

#endif // SPIRA_ARCH_X86

#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)

    // ========================================================================
    // ARM detection
    // ========================================================================

    void detect_arm() {
#if defined(__APPLE__)
        detect_arm_apple();
#elif defined(__linux__)
        detect_arm_linux();
#elif defined(_MSC_VER)
        detect_arm_windows();
#endif
    }

    void detect_cache_arm() {
#if defined(__APPLE__)
        detect_cache_arm_apple();
#elif defined(__linux__)
        detect_cache_arm_linux();
#elif defined(_MSC_VER)
        detect_cache_arm_windows();
#endif
    }

#if defined(__APPLE__)
    void detect_arm_apple() {
        // All Apple Silicon has NEON
        neon = true;

        // Check SVE via sysctl (Apple Silicon doesn't have SVE as of M4,
        // but future-proof the check)
        int64_t val = 0;
        size_t size = sizeof(val);
        if (sysctlbyname("hw.optional.arm.FEAT_SVE", &val, &size, nullptr, 0) == 0) {
            sve = (val != 0);
        }
        if (sysctlbyname("hw.optional.arm.FEAT_SVE2", &val, &size, nullptr, 0) == 0) {
            sve2 = (val != 0);
        }
    }

    void detect_cache_arm_apple() {
        auto sysctl_u32 = [](const char* name) -> uint32_t {
            uint64_t val = 0;
            size_t sz = sizeof(val);
            if (sysctlbyname(name, &val, &sz, nullptr, 0) == 0)
                return static_cast<uint32_t>(val);
            return 0;
        };
        cache.l1d_size  = sysctl_u32("hw.l1dcachesize");
        cache.l1i_size  = sysctl_u32("hw.l1icachesize");
        cache.l2_size   = sysctl_u32("hw.l2cachesize");
        cache.l3_size   = sysctl_u32("hw.l3cachesize");
        cache.line_size = sysctl_u32("hw.cachelinesize");
    }
#endif

#if defined(__linux__)
    void detect_arm_linux() {
        unsigned long hwcap = getauxval(AT_HWCAP);

#if defined(SPIRA_ARCH_ARM64)
        // AArch64: NEON is mandatory
        neon = true;

// SVE: HWCAP_SVE is bit 22 on aarch64
#if defined(HWCAP_SVE)
        sve = (hwcap & HWCAP_SVE) != 0;
#endif

        unsigned long hwcap2 = getauxval(AT_HWCAP2);
#if defined(HWCAP2_SVE2)
        sve2 = (hwcap2 & HWCAP2_SVE2) != 0;
#endif

#elif defined(SPIRA_ARCH_ARM32)
// 32-bit ARM: check NEON bit
#if defined(HWCAP_NEON)
        neon = (hwcap & HWCAP_NEON) != 0;
#endif
        // No SVE on 32-bit ARM
#endif
    }

    void detect_cache_arm_linux() {
        // Read cache topology from sysfs.
        // Each index under /sys/devices/system/cpu/cpu0/cache/ is one level.
        for (int idx = 0; idx < 16; ++idx) {
            char path[128];

            // Cache type: "Data", "Instruction", or "Unified"
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
            FILE* f = fopen(path, "r");
            if (!f) break;
            char type[32] = {};
            bool ok = (fscanf(f, "%31s", type) == 1);
            fclose(f);
            if (!ok) break;

            // Cache level (1, 2, 3, ...)
            int level = 0;
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
            f = fopen(path, "r");
            if (!f) break;
            fscanf(f, "%d", &level);
            fclose(f);

            // Cache size, e.g. "32K", "512K", "8192K"
            uint32_t sz = 0;
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
            f = fopen(path, "r");
            if (f) {
                char unit = 'K';
                if (fscanf(f, "%u%c", &sz, &unit) >= 1) {
                    if      (unit == 'M' || unit == 'm') sz *= 1024u * 1024u;
                    else                                 sz *= 1024u; // K (default)
                }
                fclose(f);
            }

            // Cache line size (same for all levels on most CPUs; read once)
            if (cache.line_size == 0) {
                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size",
                         idx);
                f = fopen(path, "r");
                if (f) { fscanf(f, "%u", &cache.line_size); fclose(f); }
            }

            bool is_data  = (strcmp(type, "Data")        == 0 ||
                             strcmp(type, "Unified")      == 0);
            bool is_instr = (strcmp(type, "Instruction") == 0 ||
                             strcmp(type, "Unified")      == 0);

            if (level == 1) {
                if (is_data)  cache.l1d_size = sz;
                if (is_instr) cache.l1i_size = sz;
            } else if (level == 2) {
                cache.l2_size = sz;
            } else if (level == 3) {
                cache.l3_size = sz;
            }
        }
    }
#endif

#if defined(_MSC_VER)
    void detect_arm_windows() {
// Windows on ARM64
#if defined(SPIRA_ARCH_ARM64)
        // NEON is mandatory on ARM64
        neon = true;

        // IsProcessorFeaturePresent doesn't expose SVE yet
        // Future: check PF_ARM_SVE_INSTRUCTIONS_AVAILABLE when defined
#elif defined(SPIRA_ARCH_ARM32)
        neon = IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != 0;
#endif
    }

    void detect_cache_arm_windows() {
        DWORD buf_size = 0;
        GetLogicalProcessorInformation(nullptr, &buf_size);
        if (buf_size == 0) return;

        auto* buf = static_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(
            std::malloc(buf_size));
        if (!buf) return;

        if (GetLogicalProcessorInformation(buf, &buf_size)) {
            DWORD count = buf_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            for (DWORD i = 0; i < count; ++i) {
                if (buf[i].Relationship != RelationCache) continue;
                const auto& ci = buf[i].Cache;
                if (cache.line_size == 0)                        cache.line_size = ci.LineSize;
                if (ci.Level == 1 && ci.Type == CacheData)        cache.l1d_size  = ci.Size;
                if (ci.Level == 1 && ci.Type == CacheInstruction) cache.l1i_size  = ci.Size;
                if (ci.Level == 2)                                 cache.l2_size   = ci.Size;
                if (ci.Level == 3)                                 cache.l3_size   = ci.Size;
            }
        }
        std::free(buf);
    }
#endif

#endif // SPIRA_ARCH_ARM64 || SPIRA_ARCH_ARM32
};

} // namespace spira::kernel

#endif
