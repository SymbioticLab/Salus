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

#ifndef SALUS_SSTL_DEBUGGING_H
#define SALUS_SSTL_DEBUGGING_H

#include "utils/date.h"
#include <cstdint>

namespace sstl {
class StackSentinel
{
    static const std::size_t kBufferSize = 16 * 1024;
    uint8_t m_buffer[kBufferSize];

public:
    StackSentinel();
    ~StackSentinel();
};

#if defined(SALUS_ENABLE_TIMEOUT_WARNING)
template<typename LogFn, typename Clock = std::chrono::system_clock>
class TimeoutWarning {
public:
    using time_point_t = std::chrono::time_point<Clock>;
    using duration_t = typename Clock::duration;

    explicit TimeoutWarning(duration_t limit, LogFn&& logfn)
        : m_logfn(std::forward<LogFn>(logfn))
        , m_limit(std::move(limit))
        , m_start(Clock::now())
    {
    }

    ~TimeoutWarning()
    {
        auto dur = Clock::now() - m_start;
        if (dur > m_limit) {
            m_logfn(m_limit, dur);
        }
    }

private:
    LogFn m_logfn;
    duration_t m_limit;
    std::chrono::time_point<Clock> m_start = Clock::now();
};
#else
template<typename LogFn, typename Clock = std::chrono::system_clock>
class TimeoutWarning {
public:
    using time_point_t = std::chrono::time_point<Clock>;
    using duration_t = typename Clock::duration;

    explicit TimeoutWarning(duration_t, LogFn&&) { }
};
#endif // SALUS_ENABLE_TIMEOUT_WARNING

} // namespace sstl

#if !defined(NDEBUG)
#define STACK_SENTINEL ::sstl::StackSentinel ss
#else
#define STACK_SENTINEL (void) 0
#endif

#endif // SALUS_SSTL_DEBUGGING_H
