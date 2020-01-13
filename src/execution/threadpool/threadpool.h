/*
 * Copyright 2019 Peifeng Yu <peifeng@umich.edu>
 * 
 * This file is part of Salus
 * (see https://github.com/SymbioticLab/Salus).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef EXECUTION_THREADPOOL_H
#define EXECUTION_THREADPOOL_H

#include "utils/fixed_function.hpp"

#include <future>
#include <memory>

struct ThreadPoolOptions
{
    /**
     * Number of threads in the thread pool.
     * Use 0 for default value, which is std::thread::hardware_concurrency()
     */
    size_t numThreads = 0;

    ThreadPoolOptions &setNumThreads(size_t num)
    {
        numThreads = num;
        return *this;
    }

    /**
     * Whether allow spinning wait in worker threads for lower latency
     */
    bool allowSpinning = true;

    ThreadPoolOptions &setAllowSpinning(bool allow)
    {
        allowSpinning = allow;
        return *this;
    }

    /**
     * Times of tries for spin wait before go to wait.
     * Use -1 for default value, which is 5000 / numThreads
     */
    int spinCount = -1;

    ThreadPoolOptions &setSpinCount(int count)
    {
        spinCount = count;
        return *this;
    }

    /**
     * @brief Optional worker thread name, truncated at 16 characters.
     */
    std::string workerName = "";

    ThreadPoolOptions &setWorkerName(const std::string &name)
    {
        workerName = name;
        return *this;
    }

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

    using Closure = sstl::FixedFunction<void()>;

    /**
     * @brief Try run a closure c in thread pool.
     * @returns c itself if queue is full. Otherwise a default constructed Closure.
     */
    Closure tryRun(Closure c);

    /**
     * @brief Run the Func f in thread pool, don't care about its completion.
     * This may be more efficient, because no wrapper task for future/promise is created.
     * If the queue is full, then f is run on calling thread.
     */
    template<typename Func>
    void run(Func f)
    {
        auto c = tryRun(std::move(f));
        if (c) {
            // enqueue failed, run on current thread
            c();
        }
    }

    /**
     * @brief Post the function f to thread pool, returns with a future holding the result.
     * If the queue is full, then closure is executed on caller thread before return.
     * @returns future holding the return value of function f.
     */
    template<typename Func>
    auto post(Func f)
    {
        using R = std::invoke_result_t<Func>;
        using Task = std::packaged_task<R()>;

        Task tk(std::move(f));
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
