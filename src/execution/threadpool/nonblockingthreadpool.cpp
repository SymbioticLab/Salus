/*
 * A non-blocking thread pool implementation with optimizations:
 * - Work stealing
 * - One spinning wait thread
 * Copyright (C) 2017 Aetf <aetf@unlimitedcodeworks.xyz>
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

#include "threadpool.h"

#include "RunQueue.h"
#include "EventCount.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

using std::unique_ptr;
using std::vector;

struct Task
{
    ThreadPool::Closure c;

    Task() = default;
    explicit Task(ThreadPool::Closure &&cc) : c(std::move(cc)) {}

    Task(Task &&) = default;
    Task &operator=(Task &&) = default;

    void operator () ()
    {
        c();
    }

    operator bool() const
    {
        return c;
    }
};

ThreadPoolOptions::ThreadPoolOptions()
{
    if (numThreads == 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    // NOTE: The time spent in steal() is proportional
    // to numThreads and we assume that new work is scheduled at a
    // constant rate, so we set spinCount to 5000 / numThreads. The
    // constant was picked based on a fair dice roll, tune it.
    spinCount = allowSpinning && numThreads > 0 ? 5000 / numThreads : 0;
}

class ThreadPoolPrivate
{
    ThreadPool *const q; // not own

    using Queue = RunQueue<Task, 1024>;

public:
    ThreadPoolPrivate(ThreadPool *q, const ThreadPoolOptions &options);
    ~ThreadPoolPrivate();

    Task tryRun(Task c);
    void stop();
    void join();
    size_t numThreads() const;
    int currentThreadId() const;

private:
    struct PerThread
    {
        constexpr PerThread()
            : pool(nullptr)
            , rand(0)
            , thread_id(-1)
        {
        }
        ThreadPoolPrivate *pool; // Parent pool, or null for normal threads.
        uint64_t rand;           // Random generator state.
        int thread_id;           // Worker thread index in pool.
    };

    /**
     * Main worker thread loop.
     */
    void workerLoop(int thread_id);

    /**
     * Steal tries to steal work from other worker threads in best-effort manner.
     */
    Task steal();

    /**
     * waitForWork blocks until new work is available (returns true), or if it is
     * time to exit (returns false). Can optionally return a task to execute in t
     * (in such case t.f != nullptr on return).
     */
    bool waitForWork(EventCount::Waiter *waiter, Task *t);

    int nonEmptyQueueIndex();

    static inline PerThread *getPerThread()
    {
        static thread_local PerThread per_thread;
        return &per_thread;
    }

    static inline unsigned rand(uint64_t *state)
    {
        auto current = *state;
        // Update the internal state
        *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
        // Generate the random output (using the PCG-XSH-RS scheme)
        return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
    }

    ThreadPoolOptions m_options;
    vector<std::thread> m_threads;
    vector<Queue> m_queues;
    vector<unsigned> m_coprimes;
    vector<EventCount::Waiter> m_waiters;
    std::atomic<unsigned> m_blocked;
    std::atomic<bool> m_spinning;
    std::atomic<bool> m_done;
    std::atomic<bool> m_cancelled;
    EventCount m_ec;
};

ThreadPool::ThreadPool(const ThreadPoolOptions &options)
    : d(std::make_unique<ThreadPoolPrivate>(this, options))
{
}

ThreadPool::~ThreadPool() = default;

ThreadPool::Closure ThreadPool::tryRun(Closure c)
{
    Task t(std::move(c));
    t = d->tryRun(std::move(t));
    return std::move(t.c);
}
void ThreadPool::stop()
{
    d->stop();
}
void ThreadPool::join()
{
    d->join();
}
size_t ThreadPool::numThreads() const
{
    return d->numThreads();
}
int ThreadPool::currentThreadId() const
{
    return d->currentThreadId();
}

ThreadPoolPrivate::ThreadPoolPrivate(ThreadPool *q, const ThreadPoolOptions &options)
    : q(q)
    , m_options(options)
    // Queue is not movable or copyable, thus can only be constructed this way
    , m_queues(options.numThreads)
    // Waiter is not movable or copyable, thus can only be constructed this way
    , m_waiters(options.numThreads)
    , m_blocked(0)
    , m_spinning(false)
    , m_done(false)
    , m_cancelled(false)
    , m_ec(m_waiters)
{
    auto numThreads = m_options.numThreads;

    m_threads.reserve(numThreads);
    m_coprimes.reserve(numThreads);

    // Calculate coprimes of numThreads.
    // Coprimes are used for a random walk over all threads in Steal
    // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
    // a walk starting thread index t and calculate numThreads - 1 subsequent
    // indices as (t + coprime) % numThreads, we will cover all threads without
    // repetitions (effectively getting a pseudo-random permutation of thread
    // indices).
    for (size_t i = 1; i <= numThreads; i++) {
        unsigned a = i;
        unsigned b = numThreads;
        // If GCD(a, b) == 1, then a and b are coprimes.
        while (b != 0) {
            unsigned tmp = a;
            a = b;
            b = tmp % b;
        }
        if (a == 1) {
            m_coprimes.push_back(i);
        }
    }

    for (size_t i = 0; i < numThreads; i++) {
        m_threads.emplace_back([this, i]() { workerLoop(i); });
    }
}

Task ThreadPoolPrivate::tryRun(Task t)
{
    auto pt = getPerThread();
    if (pt->pool == this) {
        // Worker thread of this pool, push onto the thread's queue.
        t = m_queues[pt->thread_id].PushFront(std::move(t));
    } else {
        // A free-standing thread (or worker of another pool), push onto a random
        // queue.
        t = m_queues[rand(&pt->rand) % m_queues.size()].PushBack(std::move(t));
    }
    // Note: below we touch this after making w available to worker threads.
    // Strictly speaking, this can lead to a racy-use-after-free. Consider that
    // Schedule is called from a thread that is neither main thread nor a worker
    // thread of this pool. Then, execution of w directly or indirectly
    // completes overall computations, which in turn leads to destruction of
    // this. We expect that such scenario is prevented by program, that is,
    // this is kept alive while any threads can potentially be in Schedule.
    if (!t) {
        m_ec.Notify(false);
    }
    return t;
}

void ThreadPoolPrivate::stop()
{
    m_cancelled = true;
    m_done = true;

    // Wake up the threads without work to let them exit on their own.
    m_ec.Notify(true);
}

void ThreadPoolPrivate::join()
{
    for (auto &thr : m_threads) {
        if (thr.joinable()) {
            thr.join();
        }
    }
}

size_t ThreadPoolPrivate::numThreads() const
{
    return m_options.numThreads;
}

int ThreadPoolPrivate::currentThreadId() const
{
    auto pt = getPerThread();
    if (pt->pool == this) {
        return pt->thread_id;
    } else {
        return -1;
    }
}

ThreadPoolPrivate::~ThreadPoolPrivate()
{
    m_done = true;

    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    if (!m_cancelled) {
        m_ec.Notify(true);
    } else {
        // Since we were cancelled, there might be entries in the queues.
        // Empty them to prevent their destructor from asserting.
        for (auto &q : m_queues) {
            q.Flush();
        }
    }

    // Join threads explicitly to avoid destruction order issues.
    join();

    m_threads.clear();
    m_queues.clear();
}

int ThreadPoolPrivate::nonEmptyQueueIndex()
{
    auto pt = getPerThread();
    const size_t size = m_queues.size();
    unsigned r = rand(&pt->rand);
    unsigned inc = m_coprimes[r % m_coprimes.size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
        if (!m_queues[victim].Empty()) {
            return victim;
        }
        victim += inc;
        if (victim >= size) {
            victim -= size;
        }
    }
    return -1;
}

void ThreadPoolPrivate::workerLoop(int thread_id)
{
    const auto numThreads = m_options.numThreads;
    const auto spinCount = m_options.spinCount;
    const auto allowSpinning = m_options.allowSpinning;

    auto pt = getPerThread();
    pt->pool = this;
    pt->rand = std::hash<std::thread::id>()(std::this_thread::get_id());
    pt->thread_id = thread_id;
    auto &q = m_queues[thread_id];
    auto waiter = &m_waiters[thread_id];

    if (numThreads == 1) {
        // For numThreads == 1 there is no point in going through the expensive
        // steal loop. Moreover, since steal() calls PopBack() on the victim
        // queues it might reverse the order in which ops are executed compared to
        // the order in which they are scheduled, which tends to be
        // counter-productive for the types of I/O workloads the single thread
        // pools tend to be used for.
        while (!m_cancelled) {
            auto t = q.PopFront();
            for (int i = 0; i < spinCount && !t; i++) {
                if (!m_cancelled.load(std::memory_order_relaxed)) {
                    t = q.PopFront();
                }
            }
            if (!t) {
                if (!waitForWork(waiter, &t)) {
                    return;
                }
            }
            if (t) {
                t();
            }
        }
    } else {
        while (!m_cancelled) {
            auto t = q.PopFront();
            if (!t) {
                t = steal();
                if (!t) {
                    // Leave one thread spinning. This reduces latency.
                    if (allowSpinning && !m_spinning && !m_spinning.exchange(true)) {
                        for (int i = 0; i < spinCount && !t; i++) {
                            if (!m_cancelled.load(std::memory_order_relaxed)) {
                                t = steal();
                            } else {
                                return;
                            }
                        }
                        m_spinning = false;
                    }
                    if (!t) {
                        if (!waitForWork(waiter, &t)) {
                            return;
                        }
                    }
                }
            }
            if (t) {
                t();
            }
        }
    }
}

Task ThreadPoolPrivate::steal()
{
    auto pt = getPerThread();
    const size_t size = m_queues.size();
    unsigned r = rand(&pt->rand);
    unsigned inc = m_coprimes[r % m_coprimes.size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
        auto t = m_queues[victim].PopBack();
        if (t) {
            return t;
        }
        victim += inc;
        if (victim >= size) {
            victim -= size;
        }
    }
    return {};
}

bool ThreadPoolPrivate::waitForWork(EventCount::Waiter *waiter, Task *t)
{
    // We already did best-effort emptiness check in Steal, so prepare for blocking.
    m_ec.Prewait(waiter);
    // Now do a reliable emptiness check.
    int victim = nonEmptyQueueIndex();
    if (victim != -1) {
      m_ec.CancelWait(waiter);
      if (m_cancelled) {
        return false;
      } else {
        *t = m_queues[victim].PopBack();
        return true;
      }
    }
    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    m_blocked++;
    if (m_done && m_blocked == m_options.numThreads) {
      m_ec.CancelWait(waiter);
      // Almost done, but need to re-check queues.
      // Consider that all queues are empty and all worker threads are preempted
      // right after incrementing blocked_ above. Now a free-standing thread
      // submits work and calls destructor (which sets done_). If we don't
      // re-check queues, we will exit leaving the work unexecuted.
      if (nonEmptyQueueIndex() != -1) {
        // Note: we must not pop from queues before we decrement blocked_,
        // otherwise the following scenario is possible. Consider that instead
        // of checking for emptiness we popped the only element from queues.
        // Now other worker threads can start exiting, which is bad if the
        // work item submits other work. So we just check emptiness here,
        // which ensures that all worker threads exit at the same time.
        m_blocked--;
        return true;
      }
      // Reached stable termination state.
      m_ec.Notify(true);
      return false;
    }
    m_ec.CommitWait(waiter);
    m_blocked--;
    return true;
}
