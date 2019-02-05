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

#ifndef SALUS_PLATFORM_PROFILER_H
#define SALUS_PLATFORM_PROFILER_H

#include "config.h" // IWYU: keep

#if defined(WITH_GPERFTOOLS)

#include "utils/envutils.h"
#include "platform/logging.h"

#include <gperftools/profiler.h>

class ScopedProfiling
{
    bool m_enabled;
public:
    explicit ScopedProfiling(bool enabled)
        : m_enabled(enabled)
    {
        if (!enabled) return;
        const auto &profiler_output = sstl::fromEnvVarStr("SALUS_PROFILE", "/tmp/gperf.out");
        LOG(INFO) << "Running under gperftools, output: " << profiler_output;
        ProfilerStart(profiler_output);
    }

    ~ScopedProfiling()
    {
        if (!m_enabled) return;
        ProfilerStop();
    }
};
#else
class ScopedProfiling
{
public:
    explicit ScopedProfiling(bool) {};
};
#endif

#endif // SALUS_PLATFORM_PROFILER_H
