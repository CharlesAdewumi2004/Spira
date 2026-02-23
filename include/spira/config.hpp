#pragma once

#include <cstddef>
#include <stdexcept>

#include <boundcraft/searcher.hpp>

namespace spira::config
{
    struct mode_policy
    {
        std::size_t buffersize;
        std::size_t slab_merge_threshold;

        constexpr void validate_or_throw() const
        {
            if (buffersize == 0)
            {
                throw std::invalid_argument("mode_policy: buffersize must be > 0");
            }
        }
    };

    inline constexpr mode_policy spmv{
        .buffersize = 32,
        .slab_merge_threshold = 0};

    inline constexpr mode_policy balanced{
        .buffersize = 128,
        .slab_merge_threshold = 2048};

    inline constexpr mode_policy insert_heavy{
        .buffersize = 16384,
        .slab_merge_threshold = 0};

    class mode_policy_builder
    {
    public:
        static mode_policy_builder from_spmv()
        {
            return mode_policy_builder{spmv};
        }

        static mode_policy_builder from_balanced()
        {
            return mode_policy_builder{balanced};
        }

        static mode_policy_builder from_insert_heavy()
        {
            return mode_policy_builder{insert_heavy};
        }

        mode_policy_builder &buffersize(std::size_t n)
        {
            policy_.buffersize = n;
            return *this;
        }

        mode_policy_builder &slab_merge_threshold(std::size_t n)
        {
            policy_.slab_merge_threshold = n;
            return *this;
        }

        mode_policy build() const
        {
            policy_.validate_or_throw();
            return policy_;
        }

    private:
        explicit constexpr mode_policy_builder(mode_policy base) : policy_(base)
        {
        }

        mode_policy policy_;
    };

    using aos_search_policy = boundcraft::policy::hybrid<32>;
    using soa_search_policy = boundcraft::policy::hybrid<32>;

    inline constexpr std::size_t default_row_reserve_hint = 0;

}
