/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Peifeng Yu <peifeng@umich.edu>
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

#ifndef SALUS_IOTHREADPOOL_H
#define SALUS_IOTHREADPOOL_H

#include <boost/asio.hpp>
#include <boost/thread.hpp>

namespace salus {
/**
 * @brief Simple blocking IO thread pool made from boost::asio
 */
class IOThreadPoolImpl
{
    size_t m_numThreads;

    boost::asio::io_context m_context;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> m_workguard;

    boost::thread_group m_threads;

    /**
     * @brief A wrapper class that is copy-able to pass move-only objects.
     *
     * The copy constructor/assignment should not be called
     * @tparam F
     */
    template<typename F>
    struct move_wrapper : F
    {
        move_wrapper(F &&f) // NOLINT(google-explicit-constructor)
            : F(std::forward<F>(f))
        {
        }

        move_wrapper(move_wrapper &&) noexcept = default;
        move_wrapper &operator=(move_wrapper &&) noexcept = default;

        // The following are intentially unimplemented
        // as an assert that they are not called.
        move_wrapper(const move_wrapper &);
        move_wrapper &operator=(const move_wrapper &);
    };

    template<typename T>
    auto move_handler(T &&t) -> move_wrapper<typename std::decay<T>::type>
    {
        return std::forward<T>(t);
    }

public:
    IOThreadPoolImpl();
    ~IOThreadPoolImpl();

    /*
    template<typename Func, typename SFINAE = std::enable_if<std::is_copy_constructible_v<Func> ||
    !use_moveonly_trick>> auto post(Func &&f)
    {
        return boost::asio::post(std::forward<Func>(f));
    }
    */

    template<typename Func, bool use_moveonly_trick = false>
    auto post(Func &&f)
    {
        if constexpr (!std::is_copy_constructible_v<Func> && use_moveonly_trick) {
            return boost::asio::post(move_handler(f));
        } else {
            return boost::asio::post(std::forward<Func>(f));
        }
    }

    template<typename Func>
    auto defer(Func &&f)
    {
        return boost::asio::defer(std::forward<Func>(f));
    }

private:
    void workerLoop();
};

using IOThreadPool = IOThreadPoolImpl;

} // namespace salus

#endif // SALUS_IOTHREADPOOL_H
