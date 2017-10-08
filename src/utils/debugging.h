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
 */

#ifndef UTILS_DEBUGGING_H
#define UTILS_DEBUGGING_H

#include <cstdint>

namespace utils {
class StackSentinel
{
    static const std::size_t kBufferSize = 16 * 1024;
    uint8_t m_buffer[kBufferSize];

public:
    StackSentinel();
    ~StackSentinel();
};


} // namespace utils

#if !defined(NDEBUG)
#define STACK_SENTINEL ::utils::StackSentinel ss
#else
#define STACK_SENTINEL (void)0
#endif

#endif // UTILS_DEBUGGING_H
