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

#ifndef SYMBIOTIC_SALUS_IOTHREADPOOL_H
#define SYMBIOTIC_SALUS_IOTHREADPOOL_H

#include <boost/asio.hpp>
#include <boost/thread.hpp>

namespace symbiotic::salus {
/**
 * @brief Simple blocking IO thread pool made from boost::asio
 */
template<bool use_moveonly_trick = false>
class IOThreadPool
{
    size_t m_numThreads;

    boost::asio::io_context m_context;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> m_workguard;

    boost::thread_group m_threads;

    template <typename F>
    struct move_wrapper : F
    {
        move_wrapper(F&& f) : F(std::move(f)) {}

        move_wrapper(move_wrapper&&) = default;
        move_wrapper& operator=(move_wrapper&&) = default;

        move_wrapper(const move_wrapper&);
        move_wrapper& operator=(const move_wrapper&);
    };

    template <typename T>
    auto move_handler(T&& t) -> move_wrapper<typename std::decay<T>::type>
    {
        return std::move(t);
    }

public:
    IOThreadPool();
    ~IOThreadPool();

    template<typename Func, typename SFINAE = std::enable_if<std::is_copy_constructible_v<Func> || !use_moveonly_trick>>
    auto post(Func &&f)
    {
        return boost::asio::post(std::forward<Func>(f));
    }

    template<typename Func, typename SFINAE = std::enable_if<!std::is_copy_constructible_v<Func> && use_moveonly_trick>>
    auto post(Func &&f)
    {
        static_assert(use_moveonly_trick);
        return boost::asio::post(move_handler(f));
    }

    template<typename Func>
    auto defer(Func &&f)
    {
        return boost::asio::defer(std::forward<Func>(f));
    }

private:
    void workerLoop();
};

} // namespace symbiotic::salus

#endif // SYMBIOTIC_SALUS_IOTHREADPOOL_H
