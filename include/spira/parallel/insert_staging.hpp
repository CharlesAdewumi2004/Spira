#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include <spira/config.hpp>
#include <spira/concepts.hpp>

namespace spira::parallel
{

    // ─────────────────────────────────────────────────────────────────────────────
    // Insert staging for parallel_matrix
    //
    //   direct — no_staging: an empty type; insert() writes straight to the
    //            target partition's row buffer.
    //   staged — staged_inserts: one small staging array per partition, owned by
    //            the inserting thread. insert() appends to it; when it reaches
    //            StagingN entries it is burst-flushed into the partition's rows,
    //            and lock() flushes the remainder. Writing one partition at a
    //            time keeps the active array hot in L1/L2 under random row order.
    // ─────────────────────────────────────────────────────────────────────────────

    struct no_staging
    {
    };

    template <concepts::Indexable I, concepts::Valueable V, std::size_t StagingN>
    struct staged_inserts
    {
        static_assert(StagingN >= 1, "StagingN must be >= 1 for staged insert policy");

        struct entry
        {
            std::size_t local_row;
            I           col;
            bool        add; // apply via row::add() instead of insert(); sits in
                             // the padding after col, so entry size is unchanged
            V           val;
        };

        std::vector<std::vector<entry>> bufs_;

        void init(std::size_t n_parts)
        {
            bufs_.assign(n_parts, {});
            for (auto &b : bufs_)
                b.reserve(StagingN);
        }

        // Append to partition t's staging array, flushing it when full.
        template <class Partition>
        void push(std::size_t t, Partition &p, const entry &e)
        {
            bufs_[t].push_back(e);
            if (bufs_[t].size() >= StagingN)
                flush(t, p);
        }

        // Burst-flush partition t's staging array into its row buffers.
        template <class Partition>
        void flush(std::size_t t, Partition &p)
        {
            for (const auto &e : bufs_[t])
            {
                auto &row = p.writable_row(e.local_row);
                if (e.add)
                    row.add(e.col, e.val);
                else
                    row.insert(e.col, e.val);
            }
            bufs_[t].clear();
        }

        template <class Parts>
        void flush_all(Parts &parts)
        {
            for (std::size_t t = 0; t < parts.size(); ++t)
                flush(t, parts[t]);
        }

        void clear() noexcept
        {
            for (auto &b : bufs_)
                b.clear();
        }
    };

    template <config::insert_policy IP, concepts::Indexable I, concepts::Valueable V,
              std::size_t StagingN>
    using insert_staging = std::conditional_t<IP == config::insert_policy::staged,
                                              staged_inserts<I, V, StagingN>, no_staging>;

} // namespace spira::parallel
