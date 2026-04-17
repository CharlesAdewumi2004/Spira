#pragma once

#include <atomic>
#include <condition_variable>
#include <stdexcept>
#include <cstddef>
#include <functional>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>

namespace spira::parallel
{

    // ─────────────────────────────────────────────────────────────────────────────
    // thread_pool
    //
    // Fixed-size pool of n worker jthreads. The only entry point is execute(fn),
    // which broadcasts fn to all workers and blocks until every thread completes.
    //
    // Worker lifecycle:
    //   - On construction, each worker starts and waits for work (generation counter).
    //   - execute() sets fn, bumps the generation, wakes all workers, then waits on
    //     a binary semaphore. The last worker to finish releases the semaphore.
    //   - On destruction, stop_ is set and all workers are woken; jthread destructors
    //     join automatically.
    //
    // Constraints:
    //   - n_threads >= 1 is required.
    //   - execute() is not re-entrant: do not call it from inside fn.
    //   - Not copyable or moveable (owns jthreads).
    // ─────────────────────────────────────────────────────────────────────────────

    class thread_pool
    {
    public:
        explicit thread_pool(std::size_t n_threads);
        ~thread_pool();

        thread_pool(const thread_pool &) = delete;
        thread_pool &operator=(const thread_pool &) = delete;
        thread_pool(thread_pool &&) = delete;
        thread_pool &operator=(thread_pool &&) = delete;

        // Wake all n workers, each calls fn(thread_id) where thread_id ∈ [0, size()).
        // Blocks until all threads have returned from fn.
        void execute(std::move_only_function<void(std::size_t)> fn);

        [[nodiscard]] std::size_t size() const noexcept { return n_; }

    private:
        void worker(std::size_t id);

        std::size_t n_;

        // Current job — written by execute() under start_mtx_, read by workers
        // after waking (happens-before via the mutex release/acquire pair).
        std::move_only_function<void(std::size_t)> fn_;

        // Generation counter: execute() increments it; workers detect new work by
        // comparing their local copy against this value.
        std::size_t generation_{0};
        bool stop_{false};

        std::mutex start_mtx_;
        std::condition_variable start_cv_;

        // Completion tracking: each worker increments finished_; the last one
        // releases done_ so execute() can return.
        std::atomic<std::size_t> finished_{0};
        std::binary_semaphore done_{0};

        std::vector<std::jthread> threads_;
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // Implementation
    // ─────────────────────────────────────────────────────────────────────────────

    inline thread_pool::thread_pool(std::size_t n_threads)
        : n_{n_threads}, done_{0}
    {
        if (n_threads < 1)
            throw std::invalid_argument("thread_pool requires at least one thread");
        threads_.reserve(n_threads);
        for (std::size_t i = 0; i < n_threads; ++i)
            threads_.emplace_back([this, i]
                                  { worker(i); });
    }

    inline thread_pool::~thread_pool()
    {
        {
            std::lock_guard lk(start_mtx_);
            stop_ = true;
        }
        start_cv_.notify_all();
        // jthread destructors call request_stop() and join() automatically.
    }

    inline void thread_pool::execute(std::move_only_function<void(std::size_t)> fn)
    {
        {
            std::lock_guard lk(start_mtx_);
            fn_ = std::move(fn);
            finished_.store(0, std::memory_order_relaxed);
            ++generation_;
        }
        start_cv_.notify_all();
        done_.acquire(); // block until the last worker signals completion
    }

    inline void thread_pool::worker(std::size_t id)
    {
        std::size_t local_gen = 0;
        while (true)
        {
            {
                std::unique_lock lk(start_mtx_);
                start_cv_.wait(lk, [&]
                               { return generation_ != local_gen || stop_; });
                if (stop_)
                    return;
                local_gen = generation_;
                // fn_ is visible here: written under the same mutex before notify.
            }

            fn_(id); // execute outside the lock — no re-entrancy, no contention

            if (finished_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_)
                done_.release();
        }
    }

} // namespace spira::parallel
