/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017  Aetf <aetf@unlimitedcodeworks.xyz>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EXECUTION_THREADPOOL_H
#define EXECUTION_THREADPOOL_H

#include "fixed_function.hpp"

#include <future>
#include <memory>

struct ThreadPoolOptions
{
    /**
     * Number of threads in the thread pool.
     * Use 0 for default value, which is std::thread::hardware_concurrency()
     */
    size_t numThreads = 0;

    /**
     * Whether allow spinning wait in worker threads for lower latency
     */
    bool allowSpinning = true;

    /**
     * Times of tries for spin wait before go to wait.
     * Use -1 for default value, * which is 5000 / numThreads
     */
    int spinCount = -1;

    ThreadPoolOptions();
    ThreadPoolOptions(const ThreadPoolOptions &) = default;
    ThreadPoolOptions(ThreadPoolOptions &&) = default;
};

/**
 * Thread pool extracted from Eigen
 */
class ThreadPoolPrivate;
class ThreadPool
{
    ThreadPool(const ThreadPool &other) = delete;
    ThreadPool &operator=(const ThreadPool &other) = delete;

public:
    explicit ThreadPool(const ThreadPoolOptions &options = {});
    ~ThreadPool();

    // Move-only semantic
    ThreadPool(ThreadPool &&other) = default;
    ThreadPool &operator=(ThreadPool &&other) = default;

    using Closure = tp::FixedFunction<void()>;

    /**
     * @brief Try run a closure c in thread pool.
     * @returns c itself if queue is full. Otherwise a default constructed Closure.
     */
    Closure tryRun(Closure c);

    /**
     * @brief Run the closure c in thread pool, don't care about its completion.
     * This may be more efficient, because no wrapper task for future/promise is created.
     * If the queue is full, then c is run on calling thread.
     */
    void run(Closure c);

    /**
     * @brief Post the closure c to thread pool, returns with a future holding the result.
     * If the queue is full, then closure is executed on caller thread before return.
     * @returns future holding the return value of closure c.
     */
    template<typename Closure>
    auto post(Closure &&c)
    {
        using R = std::invoke_result_t<Closure>;
        using Task = std::packaged_task<R()>;

        Task tk(std::move(c));
        auto fu = tk.get_future();
        run(std::move(tk));
        return fu;
    }

    /**
     * @brief Signal to stop the thread pool, currently running tasks will continue to run.
     */
    void stop();

    /**
     * @brief Wait for all thread to exit
     */
    void join();

    /**
     * @returns the number of threads in the pool
     */
    size_t numThreads() const;

    /**
     * @returns a logical thread index between 0 and numThreads() - 1 if called
     * from one of the threads in the pool. Returns -1 otherwise.
     */
    int currentThreadId() const;

private:
    std::unique_ptr<ThreadPoolPrivate> d;
};

#endif // EXECUTION_THREADPOOL_H
