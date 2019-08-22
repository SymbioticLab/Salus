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

#include "oplibraries/tensorflow/device/gpu/lane/lanemgr.h"

#include "oplibraries/tensorflow/device/cpu.h"
#include "oplibraries/tensorflow/device/gpu/gpu.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/tfinstance.h"
#include "utils/envutils.h"
#include "utils/threadutils.h"

namespace tfgpu = perftools::gputools;

namespace salus::oplib::tensorflow {

LaneMgr::LaneMgr()
{
    SALUS_THROW_IF_ERROR(tf::ValidateGPUMachineManager());
    auto gpu_manager = tf::GPUMachineManager();

    // Initialize GPU CUDA context
    const auto &validIds = getValidGpuIds();
    CHECK(!validIds.empty()) << "At least 1 GPU should be present";

    // Initialize CUDA runtime on each GPU
    for (auto gpuId : validIds) {
        auto se = gpu_manager->ExecutorForDevice(gpuId).ValueOrDie();

        if (!m_cpuCudaHostAlloc) {
            // Find the first valid StreamExecutor to request CUDA host memory
            // through, since any will work.
            createCudaHostAllocator(se);
        }

        tf::int64 availableMemory, totalMemory;
        if (!se->DeviceMemoryUsage(&availableMemory, &totalMemory)) {
            throw TFException(tf::errors::Unknown("Failed to query available memory for GPU ", gpuId));
        }
        availableMemory = availableMemory - 300 * 1024 * 1024;

        // We don't care about the theorical totalMemory, but what is available to us in maximum
        totalMemory = availableMemory;

        auto &gcb = m_gpus.emplace_back(*this, m_gpus.size(), gpuId, *se, totalMemory);
        gcb.availableMemory = static_cast<size_t>(availableMemory);
        CHECK_LE(gcb.availableMemory, gcb.totalMemory);
    }

    // Initialize CPU device
    auto name = tf::strings::StrCat(TFInstance::namePrefix(), "/device:CPU:0");
    // use tf::cpu_allocator to select from cpu allocatory registary
    tf::SessionOptions opt;
    tf::DeviceLocality locality;
    m_cpu = std::make_unique<SalusCPUDevice>(opt, name, tf::Bytes(256 << 20), locality, tf::cpu_allocator(),
                                             m_cpuCudaHostAlloc.get());

    // Check env
    setDisabled(sstl::fromEnvVar("SALUS_DISABLE_LANEMGR", false));
}

LaneMgr::~LaneMgr() = default;

std::vector<int> LaneMgr::getValidGpuIds()
{
    return {0};
}

tf::Device *LaneMgr::compatibleCPUDevice() const
{
    return m_cpu.get();
}

void LaneMgr::createCudaHostAllocator(tfgpu::StreamExecutor *se)
{
    struct CudaHostAllocTag;
    // 64 GB max by default
    size_t cuda_host_mem_limit_in_mb =
        sstl::fromEnvVarCached<CudaHostAllocTag>("TF_CUDA_HOST_MEM_LIMIT_IN_MB", 1_sz << 16);
    size_t cuda_host_mem_limit = cuda_host_mem_limit_in_mb * (1LL << 20);
    m_cpuCudaHostAlloc = std::make_unique<tf::BFCAllocator>(new tf::CUDAHostAllocator(se), cuda_host_mem_limit,
                                                            true /*allow_growth*/, "cuda_host_bfc" /*name*/);
}

void LaneMgr::requestLanes(Layout layout, bool isInference, RequestLaneCallback &&cb)
{
    CHECK_EQ(layout.persistentOccupation.size(), layout.memoryLimits.size());

    static bool newLaneInitialized = false;
    if (m_disabled && !newLaneInitialized) {
        newLaneInitialized = true;
        auto &gcb = m_gpus.at(0);
        gcb.newLane(gcb.availableMemory, false, sstl::with_guard(*gcb.mu));
    }

    for (size_t i = 0; i != layout.memoryLimits.size(); ++i) {
        CHECK_LE(layout.persistentOccupation.at(i), layout.memoryLimits.at(i));
    }

    auto g = sstl::with_guard(m_mu);
    m_pending.emplace_back(std::move(layout), isInference, std::move(cb));
    processRequests(std::move(g));
}

void LaneMgr::processRequests()
{
    processRequests(sstl::with_guard(m_mu));
}

void LaneMgr::processRequests(sstl::detail::Guard &&g)
{
    UNUSED(g);

    auto it = m_pending.begin();
    auto end = m_pending.end();
    while (it != end) {
        auto &req = *it;

        // TODO: the algorithm below assumes single GPU, to scale to multiple ones, a global lock is needed
        CHECK_EQ(req.layout.memoryLimits.size(), 1_sz) << "Only single lane layout is supported";

        std::vector<std::shared_ptr<LaneHolder>> lanes;
        auto &gcb = m_gpus.at(0);

        // using a best fit policy
        const auto firstIdx = 0;
        std::shared_ptr<LaneHolder> lane;
        if (req.isInference) {
            lane = gcb.bestFitForInference(req.layout.memoryLimits.at(firstIdx), req.layout.persistentOccupation.at(firstIdx));
        } else {
            lane = gcb.bestFitFor(req.layout.memoryLimits.at(firstIdx), req.layout.persistentOccupation.at(firstIdx));
        }
        if (!lane) {
            // can't find a suitable allocation.
            ++it;
            continue;
        }
        lanes.emplace_back(std::move(lane));

        req.cb(std::move(lanes));

        it = m_pending.erase(it);
    }
}

std::unique_ptr<LaneHolder> LaneMgr::GpuControlBlock::bestFitFor(size_t memory, size_t persistentSize)
{
    CHECK_GE(memory, persistentSize);

    size_t temporaryPeak = memory - persistentSize;

    auto g = sstl::with_guard(*mu);
    // first see if open a new lane is possible
    LOG(INFO) << "Checking to create lane for memory size " << memory << " available now " << availableMemory;
    if (!mgr.m_disabled) {
        if (availableMemory >= memory) {
            auto lane = newLane(memory, false, std::move(g));
            auto holder = lane->tryFit(persistentSize, temporaryPeak);
            CHECK_NE(holder, nullptr);
            return holder;
        }
    }

    struct NoSharedLaneTag
    {
    };
    if (!sstl::fromEnvVarCached<NoSharedLaneTag>("SALUS_DISABLE_SHARED_LANE", false)) {
        // use linear search because we at most will have handful of lanes
        for (auto &lane : lanes) {
            // if it is an inference lane, skip it
            if (lane->getInference())
                continue;
            if (auto holder = lane->tryFit(persistentSize, temporaryPeak)) {
                return holder;
            }
        }
    }
    // TODO: defragmentation
    return {};
}


std::unique_ptr<LaneHolder> LaneMgr::GpuControlBlock::bestFitForInference(size_t memory, size_t persistentSize)
{
    CHECK_GE(memory, persistentSize);

    size_t temporaryPeak = memory - persistentSize;

    auto g = sstl::with_guard(*mu);
    // first see if open a new lane is possible
    LOG(INFO) << "Checking to create lane for memory size " << memory << " available now " << availableMemory;
    if (!mgr.m_disabled) {
        if (availableMemory >= memory) {
            auto lane = newLane(memory, true, std::move(g));
            auto holder = lane->tryFit(persistentSize, temporaryPeak);
            CHECK_NE(holder, nullptr);
            return holder;
        }
    }

    // TODO: defragmentation
    return {};
}

sstl::ScopedUnref<GpuLane> LaneMgr::GpuControlBlock::newLane(size_t memory, bool flag, sstl::detail::Guard &&g)
{
    CHECK_GT(memory, 0);

    UNUSED(g);

    if (availableMemory < memory) {
        return sstl::ScopedUnref<GpuLane>{};
    }

    availableMemory -= memory;

    auto lane = sstl::make_scoped_unref<GpuLane>(*this, memory, nextStream++, flag);

    // Insert into lanes, which is from small to large
    auto it = lanes.begin();
    while (it != lanes.end()) {
        if ((*it)->availableMemory() > lane->availableMemory()) {
            break;
        }
        ++it;
    }
    lanes.insert(it, sstl::add_ref(lane.get()));
    return lane;
}

void LaneMgr::GpuControlBlock::removingLane(sstl::ScopedUnref<GpuLane> &&lane)
{
    auto theLane = lane.release();
    // This shouldn't be the last ref, because this->lanes holds another one
    CHECK(!theLane->Unref());

    // This is the only ref, so remove it from the list.
    if (theLane->RefCountIsOne()) {
        maybeRemoveLane(theLane);
    }

    // NOTE: may block or take long time
    mgr.processRequests();
}

void LaneMgr::GpuControlBlock::maybeRemoveLane(sstl::not_null<GpuLane *> lane)
{
    if (mgr.m_disabled) {
        return;
    }

    auto g = sstl::with_guard(*mu);

    // lanes contains ScopedUnref, so removing from
    // list calls Unref
    size_t avail = 0;
    lanes.remove_if([&](auto &wl) {
        if (wl.get() == lane) {
            avail = lane->availableMemory();
            return true;
        }
        return false;
    });

    if (avail == 0) {
        return;
    }
    availableMemory += avail;
    CHECK_LE(availableMemory, totalMemory);
}

GpuLane::GpuLane(LaneMgr::GpuControlBlock &gcb, size_t memoryLimit, int baseStreamIndex, bool flag)
    : isInference(flag)
    , m_gcb(gcb)
    , m_totalMemory(memoryLimit)
    , m_baseStreamIndex(baseStreamIndex)
    , m_availableMemory(memoryLimit)
    , m_maxPeak()
    , m_id(++NextId)
{
    // create TFDevice
    initializeDevice();
}

namespace {

std::string GetShortDeviceDescription(int device_id, const tfgpu::DeviceDescription &desc)
{
    int cc_major;
    int cc_minor;
    if (!desc.cuda_compute_capability(&cc_major, &cc_minor)) {
        cc_major = 0;
        cc_minor = 0;
    }
    return tf::strings::StrCat("device: ", device_id, ", name: ", desc.name(), ", pci bus id: ", desc.pci_bus_id(),
                               ", compute capability: ", cc_major, ".", cc_minor);
}

} // namespace

void GpuLane::initializeDevice()
{
    const std::string name = tf::strings::StrCat(TFInstance::namePrefix(), "/device:GPU:", m_gcb.index);
    const auto &desc = m_gcb.se.GetDeviceDescription();
    int numa_node = desc.numa_node();
    if (numa_node < 0) {
        // For some reason the StreamExecutor couldn't get the NUMA
        // affinity of the GPU.  If this is not a multi-socket mobo with
        // GPUs local to different buses, it doesn't matter.  If it is, we
        // may run into trouble later with data transfer operations.  The
        // trouble may manifest as slower than expected performance, or
        // outright failures.
        LOG(INFO) << "Could not identify NUMA node of " << name
                  << ", defaulting to 0.  Your kernel may not have been built "
                  << "with NUMA support.";
        numa_node = 0;
    }

    auto allocated_bytes = static_cast<tf::Bytes>(m_availableMemory);

    // Get GPU bus_id from its reported NUMA affinity.  Because GPUs are
    // virtualized in some environments, we can't just use the GPU id.
    // NUMA locales are indexed from 0, buses are indexed from 1.
    tf::DeviceLocality dev_locality;
    dev_locality.set_bus_id(numa_node + 1);
    VLOG(2) << "GPUDevice id " << m_gcb.id << " on bus " << dev_locality.bus_id() << " numa: " << numa_node
            << " pci: " << desc.pci_bus_id();

    auto process_state = tf::ProcessState::singleton();

    auto max_streams = 1;

    tf::SessionOptions opt;
    m_dev =
        std::make_unique<SalusGPUDevice>(opt, name, allocated_bytes, dev_locality, m_gcb.id,
                                         GetShortDeviceDescription(m_gcb.id, desc), getGPUAllocator(),
                                         process_state->GetCPUAllocator(numa_node), m_gcb.cudaHostAlloc(), max_streams);
    SALUS_THROW_IF_ERROR(m_dev->Init(opt));
}

tf::Allocator *GpuLane::getGPUAllocator()
{
    struct GpuLaneTag;
    if (!m_alloc) {
        tf::GPUOptions opt;
        auto useSmallOpt = sstl::fromEnvVarCached<GpuLaneTag>("SALUS_ALLOCATOR_SMALL_OPT", false);
        m_alloc = std::make_unique<tf::GPUDoubleBFCAllocator>(m_gcb.id, m_availableMemory, opt, useSmallOpt);
    }
    return m_alloc.get();
}

std::unique_ptr<LaneHolder> GpuLane::tryFit(size_t persistent, size_t peak)
{
    auto g = sstl::with_guard(m_mu);
    auto maxPeak = peak;
    if (!m_maxPeak.empty()) {
        maxPeak = std::max(maxPeak, *m_maxPeak.cbegin());
    }
    if ((persistent + maxPeak) <= m_availableMemory) {
        addHoldUnsafe(persistent, peak);
        return std::make_unique<LaneHolder>(sstl::add_ref(this), persistent, peak);
    }
    return {};
}

LaneHolder::~LaneHolder()
{
    m_lane->removeHold(m_hold, m_peak);
    // Notify LaneMgr to unref lane
    auto l = m_lane.get();
    l->notifyGCB(std::move(m_lane));

    CHECK_EQ(m_lane.get(), nullptr);
}

void GpuLane::notifyGCB(sstl::ScopedUnref<GpuLane> &&self)
{
    m_gcb.removingLane(std::move(self));
}

GpuLane::~GpuLane()
{
    // first release all resources in the lane
    m_dev.reset();
}

} // namespace salus::oplib::tensorflow
