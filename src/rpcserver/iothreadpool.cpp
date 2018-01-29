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

#include "iothreadpool.h"

#include <thread>

namespace symbiotic::salus {

IOThreadPool::IOThreadPool()
    : m_numThreads(std::max(std::hardware_concurrency() / 2, 1))
    , m_context(m_numThreads)
    , m_workguard(boost::asio::make_work_guard(m_context))
{
    while (m_threads.size() < m_numThreads) {
        m_threads.create_thread(std::bind(&IOThreadPool::workerLoop, this));
    }
}

IOThreadPool::~IOThreadPool()
{
    m_context.stop();
    m_workguard.reset();
    m_threads.join_all();
}

void IOThreadPool::workerLoop()
{
    m_context.run();
}

} // namespace symbiotic::salus
