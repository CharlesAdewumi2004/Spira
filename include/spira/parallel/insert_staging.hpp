#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

#include <spira/config.hpp>
#include <spira/concepts.hpp>

namespace spira::parallel
{

    // ─────────────────────────────────────────────────────────────────────────────
    // insert_staging<IP, I, V, StagingN>
    //
    // Compile-time policy type that controls how parallel_matrix::insert() routes
    // data to partition row buffers.
    //
    //   direct  — empty type, zero memory, zero overhead. insert() writes straight
    //             to the target partition's row buffer. Cache-hostile under random
    //             row arrival order (main thread jumps across partition memory).
    //
    //   staged  — holds N per-partition staging arrays on the main thread. insert()
    //             appends to staging[t] (small, stays hot in L1/L2). When staging[t]
    //             reaches StagingN entries it is burst-flushed sequentially into the
    //             partition's row buffers. At lock() the remainder is flushed first.
    //             StagingN must be >= 1.
    //
    // Both specialisations expose the same interface so parallel_matrix can call
    // them unconditionally inside `if constexpr` branches.
    // ─────────────────────────────────────────────────────────────────────────────

    // Primary template — only the two specialisations below are used.
    template <config::insert_policy IP, concepts::Indexable I, concepts::Valueable V,
              std::size_t StagingN>
    struct insert_staging;

    // ─────────────────────────────────────────────────────────────────────────────
    // direct specialisation — empty, no storage
    // ─────────────────────────────────────────────────────────────────────────────

    template <concepts::Indexable I, concepts::Valueable V, std::size_t StagingN>
    struct insert_staging<config::insert_policy::direct, I, V, StagingN>
    {
        void init(std::size_t /*n_parts*/) noexcept {}

        template <class Partition>
        void flush(std::size_t /*t*/, Partition & /*p*/) noexcept {}

        template <class Parts>
        void flush_all(Parts & /*parts*/) noexcept {}
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // staged specialisation — per-partition staging arrays
    // ─────────────────────────────────────────────────────────────────────────────

    template <concepts::Indexable I, concepts::Valueable V, std::size_t StagingN>
    struct insert_staging<config::insert_policy::staged, I, V, StagingN>
    {
        static_assert(StagingN >= 1, "StagingN must be >= 1 for staged insert policy");

        struct entry
        {
            std::size_t local_row;
            I           col;
            V           val;
        };

        // One staging array per partition, owned by the main (inserting) thread.
        std::vector<std::vector<entry>> bufs_;

        void init(std::size_t n_parts)
        {
            bufs_.resize(n_parts);
            for (auto &b : bufs_)
                b.reserve(StagingN);
        }

        // Burst-flush partition t's staging array into its row buffers, then clear.
        template <class Partition>
        void flush(std::size_t t, Partition &p)
        {
            for (const auto &e : bufs_[t])
                p.rows[e.local_row].insert(e.col, e.val);
            bufs_[t].clear();
        }

        // Flush all partitions — called at lock() before the pool runs.
        template <class Parts>
        void flush_all(Parts &parts)
        {
            for (std::size_t t = 0; t < parts.size(); ++t)
                flush(t, parts[t]);
        }
    };

} // namespace spira::parallel
