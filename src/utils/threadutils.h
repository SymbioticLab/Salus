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
 *
 */

#ifndef THREADUTILS_H
#define THREADUTILS_H

#include "platform/logging.h"

#include <boost/iterator/indirect_iterator.hpp>
#include <boost/thread/lock_algorithms.hpp>
#include <boost/thread/lockable_traits.hpp>
#include <boost/thread/shared_mutex.hpp>

#include <condition_variable>
#include <chrono>
#include <iterator>
#include <mutex>
#include <memory>
#include <type_traits>

namespace utils {

using Guard = std::lock_guard<std::mutex>;
using UGuard = std::unique_lock<std::mutex>;

class TGuard
{
    std::chrono::steady_clock::time_point prelock;
    UGuard g;
    std::chrono::steady_clock::time_point locked;
    std::chrono::steady_clock::time_point released;
    std::string name;

    TGuard(const TGuard &other) = delete;
    TGuard &operator = (const TGuard &other) = delete;
public:
    explicit TGuard(std::mutex &mu, const std::string &name)
        : prelock(std::chrono::steady_clock::now())
        , g(mu)
        , locked(std::chrono::steady_clock::now())
        , name(name)
    {
    }

    TGuard(TGuard &&other) = default;
    TGuard &operator = (TGuard &&other) = default;

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
            PerfLog(INFO) << "Mutex " << name << "@" << as_hex(g.mutex())
            << " usage: acquiring " << duration_cast<microseconds>(locked - prelock).count()
            << "us, locking " << duration_cast<microseconds>(released - locked).count() << "us";
        }
    }

    ~TGuard()
    {
        unlock();
    }
};

// Catch bug where variable name is omitted, e.g. Guard (mu);
#define Guard(x) static_assert(0, "Guard declaration missing variable name");
#define UGuard(x) static_assert(0, "UGuard declaration missing variable name");

template<typename Iterator>
typename std::enable_if_t<
    std::is_pointer<typename std::iterator_traits<Iterator>::value_type>::value
> lock(Iterator begin, Iterator end)
{
    boost::lock(boost::make_indirect_iterator(begin),
                boost::make_indirect_iterator(end));
}

template<typename Iterator>
typename std::enable_if_t<
    std::is_pointer<typename std::iterator_traits<Iterator>::value_type>::value
> lock_shared(Iterator begin, Iterator end);

template<typename Iterator>
typename std::enable_if_t<
    std::is_same<typename std::iterator_traits<Iterator>::value_type, boost::shared_mutex>::value
> lock_shared(Iterator begin, Iterator end);

template<typename SharedLockable>
class shared_mutex_adapter
{
public:
    shared_mutex_adapter(SharedLockable &mu) : m_mu(&mu) {}

    void lock() { m_mu->lock_shared(); }
    void unlock() noexcept { m_mu->unlock_shared(); }
    bool try_lock() noexcept { return m_mu->try_lock_shared(); }

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
    uint32_t m_count = 0; // Initialized as locked.

public:
    void notify(uint32_t c = 1);

    void wait(uint32_t c = 1);
};

/**
 * Notification that is sticky.
 */
class notification {
    std::mutex m_mu;
    std::condition_variable m_cv;
    bool m_notified = false;

public:
    void notify();
    bool notified();
    void wait();
};

} // namespace utils

namespace boost {
namespace sync {
template<typename SharedLockable>
class is_basic_lockable<utils::shared_mutex_adapter<SharedLockable>> : std::true_type {};

template<typename SharedLockable>
class is_lockable<utils::shared_mutex_adapter<SharedLockable>> : std::true_type {};
} // namespace sync
} // namespace boost

namespace utils {

template<typename Iterator>
typename std::enable_if_t<
    std::is_pointer<typename std::iterator_traits<Iterator>::value_type>::value
> lock_shared(Iterator begin, Iterator end)
{
    using PMutex = typename std::iterator_traits<Iterator>::value_type;
    using Mutex = typename std::pointer_traits<PMutex>::element_type;

    boost::indirect_iterator<Iterator, Mutex> b(begin), e(end);

    lock_shared(b, e);
}

template<typename Iterator>
typename std::enable_if_t<
    std::is_same<typename std::iterator_traits<Iterator>::value_type, boost::shared_mutex>::value
> lock_shared(Iterator begin, Iterator end)
{
    using Mutex = typename std::iterator_traits<Iterator>::value_type;
    std::vector<shared_mutex_adapter<Mutex>> adapters;
    adapters.reserve(8);
    for (auto it = begin; it != end; ++it) {
        adapters.emplace_back(*it);
    }
    boost::lock(adapters.begin(), adapters.end());
}

} // namespace utils


#endif // THREADUTILS_H
