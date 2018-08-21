//
// Created by peifeng on 4/3/18.
//

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
