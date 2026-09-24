#ifndef SPIRA_KERNELS_HW_DETECT_HPP
#define SPIRA_KERNELS_HW_DETECT_HPP

#include <cstdint>

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

namespace spira::kernel
{

    // The CPU features kernel dispatch chooses between (dispatch.cpp).
    struct CpuFeatures
    {
        // x86 features
        bool sse42 = false;
        bool avx = false; // gate for avx2 / fma / avx512f: needs OS YMM support
        bool avx2 = false;
        bool fma = false;
        bool avx512f = false;

        // ARM features
        bool neon = false;

        CpuFeatures() { detect(); }

    private:
        void detect()
        {
#if defined(SPIRA_ARCH_X86)
            detect_x86();
#elif defined(SPIRA_ARCH_ARM64)
            neon = true; // mandatory on AArch64
#endif
        }

#if defined(SPIRA_ARCH_X86)

        // ========================================================================
        // x86 detection
        // ========================================================================

        struct CpuidResult
        {
            uint32_t eax, ebx, ecx, edx;
        };

        static CpuidResult cpuid(uint32_t leaf, uint32_t subleaf = 0)
        {
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

        static uint64_t xgetbv(uint32_t index)
        {
#if defined(_MSC_VER)
            return _xgetbv(index);
#else
            uint32_t eax, edx;
            __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
            return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
        }

        void detect_x86()
        {
            // Check max supported CPUID leaf
            auto leaf0 = cpuid(0);
            uint32_t max_leaf = leaf0.eax;

            if (max_leaf < 1)
                return;

            // Leaf 1: basic features
            auto leaf1 = cpuid(1);
            sse42 = (leaf1.ecx >> 20) & 1;         // ECX bit 20
            bool os_xsave = (leaf1.ecx >> 27) & 1; // ECX bit 27 — OS supports XSAVE
            avx = (leaf1.ecx >> 28) & 1;           // ECX bit 28
            fma = (leaf1.ecx >> 12) & 1;           // ECX bit 12

            // AVX/FMA require OS support for saving YMM registers
            if (avx && os_xsave)
            {
                uint64_t xcr0 = xgetbv(0);
                bool os_saves_xmm = (xcr0 & 0x2) != 0; // bit 1: XMM state
                bool os_saves_ymm = (xcr0 & 0x4) != 0; // bit 2: YMM state

                if (!os_saves_xmm || !os_saves_ymm)
                {
                    avx = false;
                    fma = false;
                }
            }
            else
            {
                // No OS XSAVE support — disable AVX and everything above
                avx = false;
                fma = false;
            }

            // Leaf 7: extended features (requires AVX to be usable)
            if (max_leaf >= 7 && avx)
            {
                auto leaf7 = cpuid(7, 0);
                avx2 = (leaf7.ebx >> 5) & 1;      // EBX bit 5
                avx512f = (leaf7.ebx >> 16) & 1;  // EBX bit 16

                // AVX-512 requires OS support for ZMM registers and opmask
                if (avx512f)
                {
                    uint64_t xcr0 = xgetbv(0);
                    bool os_saves_opmask = (xcr0 & 0x20) != 0; // bit 5
                    bool os_saves_zmm_lo = (xcr0 & 0x40) != 0; // bit 6
                    bool os_saves_zmm_hi = (xcr0 & 0x80) != 0; // bit 7

                    if (!os_saves_opmask || !os_saves_zmm_lo || !os_saves_zmm_hi)
                        avx512f = false;
                }
            }
            else
            {
                avx2 = false;
                avx512f = false;
            }
        }

#endif // SPIRA_ARCH_X86

    };

} // namespace spira::kernel

#endif
