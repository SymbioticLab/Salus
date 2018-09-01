/*
 * Copyright (c) 2018, peifeng <email>
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

#ifndef SALUS_OPLIB_TENSORFLOW_LANEMGR_H
#define SALUS_OPLIB_TENSORFLOW_LANEMGR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "oplibraries/tensorflow/tfutils.h"
#include "utils/fixed_function.hpp"
#include "utils/threadutils.h"
#include "utils/pointerutils.h"

#include <memory>
#include <list>
#include <queue>
#include <atomic>

namespace salus::oplib::tensorflow {

class SalusCPUDevice;
class GpuLane;
class LaneHolder;
class LaneMgr
{
public:
    LaneMgr();
    ~LaneMgr();

    using RequestLaneCallback = sstl::FixedFunction<void(std::vector<std::shared_ptr<LaneHolder>> &&)>;
    struct Layout
    {
        std::vector<size_t> memoryLimits;
        std::vector<size_t> persistentOccupation;
    };
    void requestLanes(Layout layout, RequestLaneCallback &&cb);

    tf::Device *compatibleCPUDevice() const;

    void setDisabled(bool value)
    {
        CHECK(value);
        m_disabled = value;
    }

private:
    std::vector<int> getValidGpuIds();
    void createCudaHostAllocator(tfgpu::StreamExecutor *se);

    bool m_disabled = false;

    struct LaneRequest
    {
        Layout layout;
        RequestLaneCallback cb;

        LaneRequest() = default;
        LaneRequest(Layout &&layout, RequestLaneCallback &&cb)
            : layout(std::move(layout))
            , cb(std::move(cb))
        {}
    };
    std::mutex m_mu;
    std::queue<LaneRequest> m_pending GUARDED_BY(m_mu);
    void processRequests();
    void processRequests(sstl::detail::Guard &&g);

    friend class GpuLane;
    class GpuControlBlock
    {
        LaneMgr &mgr;
    public:
        explicit GpuControlBlock(LaneMgr &mgr, int index, int gpuId, tfgpu::StreamExecutor &se, size_t totalMemory)
            : mgr(mgr)
              , index(index)
              , id(gpuId)
              , se(se)
              , totalMemory(totalMemory)
        {
        }

        tf::Allocator *cudaHostAlloc() const {
            return mgr.m_cpuCudaHostAlloc.get();
        }

        const int index;
        const int id;
        tfgpu::StreamExecutor &se;
        const size_t totalMemory;

        size_t availableMemory GUARDED_BY(*mu) {0};

        // Has to by dynamic allocated otherwise GCB can't be placed in vector
        std::unique_ptr<std::mutex> mu {std::make_unique<std::mutex>()};
        int nextStream GUARDED_BY(*mu) {0};

        // lanes are sorted in asc order
        std::list<sstl::ScopedUnref<GpuLane>> lanes GUARDED_BY(*mu);

        std::unique_ptr<LaneHolder> bestFitFor(size_t memory, size_t persistentSize);

        sstl::ScopedUnref<GpuLane> newLane(size_t memory, sstl::detail::Guard &&g);

        void removingLane(sstl::ScopedUnref<GpuLane> &&lane);
        void maybeRemoveLane(sstl::not_null<GpuLane *> lane);
    };
    std::vector<GpuControlBlock> m_gpus;
    std::unique_ptr<tf::Allocator> m_cpuCudaHostAlloc;
    std::unique_ptr<SalusCPUDevice> m_cpu;
};

class GpuLane : public tf::core::RefCounted
{
public:
    tf::Device *as_tfdevice() const
    {
        return m_dev.get();
    }

    size_t availableMemory() const
    {
        return m_availableMemory.load(std::memory_order_acquire);
    }

    size_t totalMemory() const
    {
        return m_totalMemory;
    }

    int baseStreamIndex() const
    {
        return m_baseStreamIndex;
    }

    /**
     * @brief Add new session here
     * @param size
     */
    void addHold(size_t size)
    {
        m_availableMemory -= size;
    }

    void removeHold(size_t size)
    {
        m_availableMemory += size;
    }

    void notifyGCB(sstl::ScopedUnref<GpuLane> &&self);

    GpuLane(LaneMgr::GpuControlBlock &gcb, size_t memoryLimit, int baseStreamIndex);
    ~GpuLane();
private:

    void initializeDevice();
    tf::Allocator *getGPUAllocator();

    LaneMgr::GpuControlBlock &m_gcb;

    const size_t m_totalMemory;
    std::atomic_uint_fast64_t m_availableMemory;
    const int m_baseStreamIndex;

    std::unique_ptr<tf::Allocator> m_alloc;
    std::unique_ptr<tf::BaseGPUDevice> m_dev;
};

class LaneHolder
{
    sstl::ScopedUnref<GpuLane> m_lane;
    size_t m_hold;
public:
    explicit LaneHolder(sstl::ScopedUnref<GpuLane> &&lane, size_t hold)
        : m_lane(std::move(lane))
        , m_hold(hold)
    {
        m_lane->addHold(m_hold);
    }

    ~LaneHolder();

    tf::Device *as_tfdevice() const
    {
        return m_lane->as_tfdevice();
    }
};

} // namespace salus::oplib::tensorflow
#endif // SALUS_OPLIB_TENSORFLOW_LANEMGR_H
