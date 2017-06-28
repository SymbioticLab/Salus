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

namespace utils {

void semaphore::notify(uint32_t c)
{
    std::unique_lock<decltype(mutex_)> lock(mutex_);
    count_ += c;
    condition_.notify_one();
}

void semaphore::wait(uint32_t c)
{
    std::unique_lock<decltype(mutex_)> lock(mutex_);
    while (count_ < c) // Handle spurious wake-ups.
        condition_.wait(lock);
    count_ -= c;
}

} // end of namespace utils
