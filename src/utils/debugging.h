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
