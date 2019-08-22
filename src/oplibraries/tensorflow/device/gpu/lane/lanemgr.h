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

#ifndef SALUS_OPLIB_TENSORFLOW_LANEMGR_H
#define SALUS_OPLIB_TENSORFLOW_LANEMGR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "oplibraries/tensorflow/tfutils.h"
#include "utils/fixed_function.hpp"
#include "utils/pointerutils.h"
#include "utils/threadutils.h"

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <set>

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
    void requestLanes(Layout layout, bool flag,  RequestLaneCallback &&cb);

    tf::Device *compatibleCPUDevice() const;

    void setDisabled(bool value)
    {
        m_disabled = value;
    }

    size_t numGPUs() const
    {
        return m_gpus.size();
    }

    size_t totalMemoryForGPU(size_t index) const
    {
        return m_gpus.at(index).totalMemory;
    }

private:
    std::vector<int> getValidGpuIds();
    void createCudaHostAllocator(tfgpu::StreamExecutor *se);

    bool m_disabled = false;

    struct LaneRequest
    {
        Layout layout;
        bool isInference;
        RequestLaneCallback cb;
        LaneRequest() = default;
        LaneRequest(Layout &&layout, bool flag, RequestLaneCallback &&cb)
            : layout(std::move(layout))
            , isInference(flag)
            , cb(std::move(cb))
        {
        }
    };
    std::mutex m_mu;
    std::list<LaneRequest> m_pending GUARDED_BY(m_mu);
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

        tf::Allocator *cudaHostAlloc() const
        {
            return mgr.m_cpuCudaHostAlloc.get();
        }

        const int index;
        const int id;
        tfgpu::StreamExecutor &se;
        const size_t totalMemory;

        size_t availableMemory GUARDED_BY(*mu){0};

        // Has to by dynamic allocated otherwise GCB can't be placed in vector
        std::unique_ptr<std::mutex> mu{std::make_unique<std::mutex>()};
        int nextStream GUARDED_BY(*mu){0};

        // lanes are sorted in asc order
        std::list<sstl::ScopedUnref<GpuLane>> lanes GUARDED_BY(*mu);

        std::unique_ptr<LaneHolder> bestFitFor(size_t memory, size_t persistentSize);
        std::unique_ptr<LaneHolder> bestFitForInference(size_t memory, size_t persistentSize);
        sstl::ScopedUnref<GpuLane> newLane(size_t memory, bool flag, sstl::detail::Guard &&g);

        void removingLane(sstl::ScopedUnref<GpuLane> &&lane);
        void maybeRemoveLane(sstl::not_null<GpuLane *> lane);
    };
    std::vector<GpuControlBlock> m_gpus;
    std::unique_ptr<tf::Allocator> m_cpuCudaHostAlloc;
    std::unique_ptr<SalusCPUDevice> m_cpu;
};

class LaneHolder;
class GpuLane : public tf::core::RefCounted
{
public:
    uint64_t id() const
    {
        return m_id;
    }

    tf::Device *as_tfdevice() const
    {
        return m_dev.get();
    }

    std::unique_ptr<LaneHolder> tryFit(size_t persistent, size_t peak);

    size_t availableMemory() const
    {
        auto g = sstl::with_guard(m_mu);
        return m_availableMemory;
    }

    size_t totalMemory() const
    {
        return m_totalMemory;
    }

    int baseStreamIndex() const
    {
        return m_baseStreamIndex;
    }

    void removeHold(size_t size, size_t peak)
    {
        auto g = sstl::with_guard(m_mu);
        m_availableMemory += size;
        auto it = m_maxPeak.find(peak);
        CHECK_NE(it, m_maxPeak.end());
        m_maxPeak.erase(it);
    }

    void notifyGCB(sstl::ScopedUnref<GpuLane> &&self);

    GpuLane(LaneMgr::GpuControlBlock &gcb, size_t memoryLimit, int baseStreamIndex, bool flag);
    ~GpuLane() override;

    bool getInference() const {
        return isInference;
    }
private:
    bool isInference;
    /**
     * @brief Add new session here
     * @param size
     */
    void addHoldUnsafe(size_t size, size_t peak)
    {
        m_availableMemory -= size;
        m_maxPeak.insert(peak);
    }

    void initializeDevice();
    tf::Allocator *getGPUAllocator();

    LaneMgr::GpuControlBlock &m_gcb;

    const size_t m_totalMemory;
    const int m_baseStreamIndex;

    mutable std::mutex m_mu;
    size_t m_availableMemory GUARDED_BY(m_mu);
    std::multiset<size_t, std::greater<>> m_maxPeak GUARDED_BY(m_mu);

    std::unique_ptr<tf::Allocator> m_alloc;
    std::unique_ptr<tf::BaseGPUDevice> m_dev;

    inline static std::atomic_uint_fast64_t NextId{0};
    uint64_t m_id;
};

class LaneHolder
{
    sstl::ScopedUnref<GpuLane> m_lane;
    size_t m_hold;
    size_t m_peak;

public:
    explicit LaneHolder(sstl::ScopedUnref<GpuLane> &&lane, size_t hold, size_t peak)
        : m_lane(std::move(lane))
        , m_hold(hold)
        , m_peak(peak)
    {
    }

    ~LaneHolder();

    tf::Device *as_tfdevice() const
    {
        return m_lane->as_tfdevice();
    }

    uint64_t id() const
    {
        return m_lane->id();
    }

    size_t availableMemory() const
    {
        return m_lane->availableMemory();
    }

    size_t totalMemory() const
    {
        return m_lane->totalMemory();
    }

    int baseStreamIndex() const
    {
        return m_lane->baseStreamIndex();
    }
};

} // namespace salus::oplib::tensorflow
#endif // SALUS_OPLIB_TENSORFLOW_LANEMGR_H
