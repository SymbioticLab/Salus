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
