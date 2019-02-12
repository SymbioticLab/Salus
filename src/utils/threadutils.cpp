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

#include "threadutils.h"

namespace sstl {

void semaphore::notify(uint32_t c)
{
    {
        auto l = with_guard(m_mu);
        m_count += c;
    }
    // Don't notify under the lock.
    m_cv.notify_all();
}

void semaphore::wait(uint32_t c)
{
    auto lock = with_uguard(m_mu);
    m_cv.wait(lock, [&]() { return m_count >= c; });
    m_count -= c;
}

void notification::notify()
{
    auto g = with_guard(m_mu);
    m_notified = true;
    m_cv.notify_all();
}

bool notification::notified()
{
    auto g = with_guard(m_mu);
    return m_notified;
}

void notification::wait()
{
    auto g = with_uguard(m_mu);
    while (!m_notified) {
        m_cv.wait(g);
    }
    m_notified = false;
}

} // namespace sstl
