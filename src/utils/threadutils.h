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

#ifndef SALUS_SSTL_THREADUTILS_H
#define SALUS_SSTL_THREADUTILS_H

#include "platform/logging.h"
#include "utils/macros.h"

#include <boost/iterator/indirect_iterator.hpp>
#include <boost/thread/lock_algorithms.hpp>
#include <boost/thread/lockable_traits.hpp>
#include <boost/thread/shared_mutex.hpp>

#include <chrono>
#include <condition_variable>
#include <iterator>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace sstl {

namespace detail {
using Guard = std::lock_guard<std::mutex>;
using UGuard = std::unique_lock<std::mutex>;
class TGuard
{
    std::chrono::steady_clock::time_point prelock;
    UGuard g;
    std::chrono::steady_clock::time_point locked;
    std::chrono::steady_clock::time_point released;
    std::string name;

public:
    SALUS_DISALLOW_COPY_AND_ASSIGN(TGuard);

    explicit TGuard(std::mutex &mu, std::string name)
        : prelock(std::chrono::steady_clock::now())
          , g(mu)
          , locked(std::chrono::steady_clock::now())
          , name(std::move(name))
    {
    }

    TGuard(TGuard &&other) = default;
    TGuard &operator=(TGuard &&other) = default;

    void lock()
    {
        prelock = std::chrono::steady_clock::now();
        g.lock();
        locked = std::chrono::steady_clock::now();
    }

    void unlock()
    {
        auto was_own = g.owns_lock();
        g.unlock();
        released = std::chrono::steady_clock::now();
        // print time usage
        if (was_own) {
            using namespace std::chrono;
            LogPerf() << "Mutex " << name << "@" << as_hex(g.mutex()) << " usage: acquiring "
                      << duration_cast<microseconds>(locked - prelock).count() << "us, locking "
                      << duration_cast<microseconds>(released - locked).count() << "us";
        }
    }

    ~TGuard()
    {
        unlock();
    }
};
} // namespace detail

template<typename ... Args>
[[nodiscard]] auto with_guard(Args && ... args)
{
    return detail::Guard(std::forward<Args>(args)...);
}

template<typename ... Args>
[[nodiscard]] auto with_uguard(Args && ... args)
{
    return detail::UGuard(std::forward<Args>(args)...);
}

template<typename ... Args>
[[nodiscard]] auto with_tguard(Args && ... args)
{
    return detail::TGuard(std::forward<Args>(args)...);
}

template<typename Iterator, typename = std::enable_if_t<
                                std::is_pointer_v<typename std::iterator_traits<Iterator>::value_type>>>
void lock(Iterator begin, Iterator end)
{
    boost::lock(boost::make_indirect_iterator(begin), boost::make_indirect_iterator(end));
}

template<typename Iterator>
std::enable_if_t<std::is_pointer_v<typename std::iterator_traits<Iterator>::value_type>> lock_shared(
    Iterator begin, Iterator end);

template<typename Iterator, typename = std::enable_if_t<std::is_same_v<
                                typename std::iterator_traits<Iterator>::value_type, boost::shared_mutex>>>
void lock_shared(Iterator begin, Iterator end);

template<typename SharedLockable>
class shared_mutex_adapter
{
public:
    explicit shared_mutex_adapter(SharedLockable &mu)
        : m_mu(&mu)
    {
    }

    void lock()
    {
        m_mu->lock_shared();
    }
    void unlock() noexcept
    {
        m_mu->unlock_shared();
    }
    bool try_lock() noexcept
    {
        return m_mu->try_lock_shared();
    }

private:
    SharedLockable *m_mu;
};

template<typename SharedLockable>
inline auto make_shared_mutex_adapter(SharedLockable &mu)
{
    return shared_mutex_adapter<SharedLockable>(mu);
}

/**
 * Semaphore that can wait on count.
 */
class semaphore
{
    std::mutex m_mu;
    std::condition_variable m_cv;
    uint64_t m_count = 0;

public:
    // Initialized as locked.
    explicit semaphore(uint64_t init = 0) : m_count(init) {}

    void notify(uint64_t c = 1);

    void wait(uint64_t c = 1);

    bool may_block(uint64_t c = 1);
};

/**
 * Notification that is sticky.
 */
class notification
{
    std::mutex m_mu;
    std::condition_variable m_cv;
    bool m_notified = false;

public:
    void notify();
    bool notified();
    void wait();
};

} // namespace sstl

namespace boost {
namespace sync {
template<typename SharedLockable>
class is_basic_lockable<sstl::shared_mutex_adapter<SharedLockable>> : std::true_type
{
};

template<typename SharedLockable>
class is_lockable<sstl::shared_mutex_adapter<SharedLockable>> : std::true_type
{
};
} // namespace sync
} // namespace boost

namespace sstl {

template<typename Iterator>
std::enable_if_t<std::is_pointer_v<typename std::iterator_traits<Iterator>::value_type>> lock_shared(
    Iterator begin, Iterator end)
{
    using PMutex = typename std::iterator_traits<Iterator>::value_type;
    using Mutex = typename std::pointer_traits<PMutex>::element_type;

    boost::indirect_iterator<Iterator, Mutex> b(begin), e(end);

    lock_shared(b, e);
}

template<typename Iterator, typename = std::enable_if_t<std::is_same_v<
                                typename std::iterator_traits<Iterator>::value_type, boost::shared_mutex>>>
void lock_shared(Iterator begin, Iterator end)
{
    using Mutex = typename std::iterator_traits<Iterator>::value_type;
    std::vector<shared_mutex_adapter<Mutex>> adapters;
    adapters.reserve(8);
    for (auto it = begin; it != end; ++it) {
        adapters.emplace_back(*it);
    }
    boost::lock(adapters.begin(), adapters.end());
}

} // namespace sstl

#endif // SALUS_SSTL_THREADUTILS_H
