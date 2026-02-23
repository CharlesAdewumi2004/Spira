#ifndef SPIRA_KERNELS_CPU_DETECT_H
#define SPIRA_KERNELS_CPU_DETECT_H

#include <cstdint>  

// ============================================================================
// Platform detection
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define SPIRA_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SPIRA_ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
#define SPIRA_ARCH_ARM32 1
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
#endif
#endif

namespace spira::kernel {

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

    CpuFeatures() { detect(); }

  private:
    void detect() {
#if defined(SPIRA_ARCH_X86)
        detect_x86();
#elif defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)
        detect_arm();
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
#endif

#endif // SPIRA_ARCH_ARM64 || SPIRA_ARCH_ARM32
};

} // namespace spira::kernel

#endif