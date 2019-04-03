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

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/v3/smblocker.h"
#include "utils/threadutils.h"
#include "utils/containerutils.h"

#include <vector>

namespace {

struct SalusCudaKernelLaunchParams
{
    struct Vec3
    {
        uint64_t x;
        uint64_t y;
        uint64_t z;
    };
    Vec3 blockCount;
    Vec3 threadPerBlock;
    uint64_t sharedMemBytes;
};

thread_local std::vector<SalusCudaKernelLaunchParams> SavedCudaKernelLaunches{};
thread_local uint64_t CurrentThreadHoldingBlocks = 0;

inline auto max(uint64_t a, SalusCudaKernelLaunchParams::Vec3 vec)
{
    auto b = vec.x * vec.y * vec.z;
    return std::max(a, b);
}

} // namespace

extern "C" {

void salus_kernel_launch_callback(unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                  unsigned int sharedMemBytes,
                                  void *)
{
    SavedCudaKernelLaunches.push_back(SalusCudaKernelLaunchParams{
        {gridDimX, gridDimY, gridDimZ},
        {blockDimX, blockDimY, blockDimZ},
        sharedMemBytes,
    });
    LOG(DEBUG) << "Got kernel launch params: ("
               << gridDimX << "," << gridDimY << "," << gridDimZ
               << ") x (" << blockDimX << "," << blockDimY << "," << blockDimZ << "), " << sharedMemBytes;
}

} // extern "C"

namespace salus::oplib::tensorflow {

SMBlocker &SMBlocker::instance()
{
    static SMBlocker blocker;
    return blocker;
}

SMUsage SMBlocker::queryAvailableSM()
{
    auto gpu_manager = tf::GPUMachineManager();
    // TODO: assume each device has the same number of SM
    auto se = gpu_manager->ExecutorForDevice(0).ValueOrDie();
    return {
        se->GetDeviceDescription().threads_per_block_limit(),
        static_cast<uint64_t>(se->GetDeviceDescription().core_count())
    };
}

SMBlocker::SMBlocker()
    : m_maxUsage{queryAvailableSM()}
    , m_freeBlocks(m_maxUsage.blockCount)
{
}

void SMBlocker::saveCurrentThreadResults(uint64_t graphId, int nodeId)
{
    // release blocks first
    {
        m_freeBlocks.notify(CurrentThreadHoldingBlocks);
        CurrentThreadHoldingBlocks = 0;
    }

    // update cache
    std::unique_lock l{m_mu};

    auto &usage = m_cache[std::make_pair(graphId, nodeId)];

    if (usage.blockCount != 0 || usage.threadPerBlock != 0) {
        LOG(WARNING) << "Overriding SM usage for graph " << graphId << " node " << nodeId
                     << ", previous: " << usage.blockCount << " " << usage.threadPerBlock;
    }

    for (const auto &res : SavedCudaKernelLaunches) {
        usage.threadPerBlock = max(usage.threadPerBlock, res.threadPerBlock);
        usage.blockCount = max(usage.blockCount, res.blockCount);
    }
    SavedCudaKernelLaunches.clear();
}

bool SMBlocker::maybeBlock(uint64_t graphId, int nodeId)
{
    auto smUsage = getUsageForKernel(graphId, nodeId);

    return m_freeBlocks.may_block(smUsage);
}

void SMBlocker::wait(uint64_t graphId, int nodeId)
{
    auto smUsage = getUsageForKernel(graphId, nodeId);

    // save the count
    CurrentThreadHoldingBlocks = smUsage;

    LOG(DEBUG) << "Wait at SMBlocker: graph " << graphId << " node " << nodeId;
    m_freeBlocks.wait(smUsage);
    LOG(DEBUG) << "Passed at SMBlocker: graph " << graphId << " node " << nodeId;
}

uint64_t SMBlocker::getUsageForKernel(uint64_t graphId, int nodeId)
{
    std::shared_lock l{m_mu};

    auto usage = sstl::getOrDefault(m_cache, {graphId, nodeId}, {});

    return std::min(usage.blockCount, m_maxUsage.blockCount);
}

} // namespace salus::oplib::tensorflow
