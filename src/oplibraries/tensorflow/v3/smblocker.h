/*
 * Copyright (c) 2019, peifeng <email>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SALUS_OPLIB_TENSORFLOW_SMBLOCKER_H
#define SALUS_OPLIB_TENSORFLOW_SMBLOCKER_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "utils/threadutils.h"

#include <boost/functional/hash.hpp>

#include <vector>
#include <unordered_map>
#include <shared_mutex>

namespace salus::oplib::tensorflow {
struct SMUsage
{
    uint64_t threadPerBlock = 0;
    uint64_t blockCount = 0;

    bool operator ==(const SMUsage &other) {
        return threadPerBlock == other.threadPerBlock && blockCount == other.blockCount;
    }

    bool operator !=(const SMUsage &other) {
        return !(*this == other);
    }
};

class SMBlocker
{
public:
    static SMBlocker &instance();

    static void setScaleFactorSM(double factor)
    {
        m_scaleFactorSM = factor;
    }

    static double scaleFactorSM()
    {
        CHECK_NE(m_scaleFactorSM, 0.0) << "Must call SMBlocker::setScaleFactorSM before getting value";
        return m_scaleFactorSM;
    }

    /**
     * @brief Release this amount of numSms
     */
    void release(uint64_t numSms);

    /**
     * @brief Return the number of sms held by current thread
     * @return
     */
    uint64_t currentThreadSMHolding() const;

    /**
     * @brief Save current thread's launch parameter
     * @param graphId
     * @param nodeId
     */
    void saveCurrentThreadResults(uint64_t graphId, int nodeId);

    /**
     * @brief Non-blocking version of wait
     * @param graphId
     * @param nodeId
     * @param priority Smaller priority is higher, default is 10
     * @return true if successfully get needed resource
     */
    bool tryTake(uint64_t graphId, int nodeId, int priority);

    /**
     * @brief Blocking wait, takes SMs
     * @param graphId
     * @param nodeId
     * @param priority
     */
    void wait(uint64_t graphId, int nodeId, int priority);

    static constexpr int MaxPriority = 100;

private:
    static double m_scaleFactorSM;
    static SMUsage queryAvailableSM();

    explicit SMBlocker(double factor);

    uint64_t getUsageForKernel(uint64_t graphId, int nodeId);

    class MaxSMUsage
    {
        SMUsage usage;
        double scale;
    public:
        explicit MaxSMUsage(SMUsage u, double scale = 1.0)
            : usage(u)
            , scale(scale)
        {}

        SMUsage get() const {
            return {usage.threadPerBlock, static_cast<uint64_t>(usage.blockCount * scale)};
        }
        double getScale() const {
            return scale;
        }
        void set(SMUsage u) {
            usage = u;
        }
        void setScale(double s) {
            scale = s;
        }
    };

    MaxSMUsage m_maxUsage;

    sstl::priority_semaphore<MaxPriority> m_freeBlocks;

    using KernelId = std::pair<uint64_t, int>;
    std::unordered_map<KernelId, SMUsage, boost::hash<KernelId>> m_cache;
    std::shared_mutex m_mu;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_SMBLOCKER_H
