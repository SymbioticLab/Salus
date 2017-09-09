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

#include <condition_variable>
#include <mutex>
#include <memory>

namespace utils {

using Guard = std::lock_guard<std::mutex>;
using UGuard = std::unique_lock<std::mutex>;

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

} // end of namespace utils


#endif // THREADUTILS_H
