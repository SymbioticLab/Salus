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

#include "iothreadpool.h"
#include "platform/thread_annotations.h"

#include <thread>

namespace salus {

IOThreadPoolImpl::IOThreadPoolImpl()
    : m_numThreads(std::max(std::thread::hardware_concurrency() / 2, 1u))
    , m_context(static_cast<int>(m_numThreads))
    , m_workguard(boost::asio::make_work_guard(m_context))
{
    while (m_threads.size() < m_numThreads) {
        m_threads.create_thread(std::bind(&IOThreadPoolImpl::workerLoop, this));
    }
}

IOThreadPoolImpl::~IOThreadPoolImpl()
{
    m_context.stop();
    m_workguard.reset();
    m_threads.join_all();
}

void IOThreadPoolImpl::workerLoop()
{
    threading::set_thread_name("salus::IOThreadPoolWorker");
    m_context.run();
}

} // namespace salus
