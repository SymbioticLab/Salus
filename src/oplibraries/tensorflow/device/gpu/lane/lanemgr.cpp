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

void LaneMgr::requestLanes(Layout layout, RequestLaneCallback &&cb)
{
    auto g = sstl::with_guard(m_mu);
    m_pending.emplace(std::move(layout), std::move(cb));
    processRequests(std::move(g));
}

void LaneMgr::processRequests()
{
    processRequests(sstl::with_guard(m_mu));
}

void LaneMgr::processRequests(sstl::detail::Guard &&g)
{
    UNUSED(g);
    while (!m_pending.empty()) {
        auto &req = m_pending.front();

        // TODO: the algorithm below assumes single GPU, to scale to multiple ones, a global lock is needed
        CHECK_EQ(req.layout.memoryLimits.size(), 1_sz) << "Only single lane layout is supported";

        std::vector<std::shared_ptr<GpuLane>> lanes;
        auto &gcb = m_gpus.at(0);

        // using a best fit policy
        auto lane = gcb.bestFitFor(req.layout.memoryLimits.at(0));
        if (!lane) {
            // can't find a suitable allocation.
            continue;
        }
        lanes.emplace_back(std::move(lane));

        req.cb(std::move(lanes));
        m_pending.pop();
    }
}

std::shared_ptr<GpuLane> LaneMgr::GpuControlBlock::bestFitFor(size_t memory)
{
    auto g = sstl::with_guard(*mu);
    // first see if open a new lane is possible
    if (availableMemory >= memory) {
        return newLane(memory, std::move(g));
    }

    // use linear search because we at most will have handful of lanes
    for (auto &wlane : lanes) {
        if (auto lane = wlane.lock()) {
            if (lane->availableMemory() >= memory) {
                return lane;
            }
        }
    }

    // TODO: extend an existing lane if all above failed
    return {};
}

std::shared_ptr<GpuLane> LaneMgr::GpuControlBlock::newLane(size_t memory, sstl::detail::Guard &&g)
{
    UNUSED(g);

    if (availableMemory < memory) {
        return {};
    }

    availableMemory -= memory;

    auto lane = std::make_shared<GpuLane>(*this, memory, nextStream++);

    // Insert into lanes, which is from small to large
    auto it = lanes.begin();
    while (it != lanes.end()) {
        auto l = it->lock();
        if (!l) {
            continue;
        }
        if (l->availableMemory() > lane->availableMemory()) {
            break;
        }
        ++it;
    }
    lanes.insert(it, lane);
    return lane;
}

void LaneMgr::GpuControlBlock::laneRemoved(size_t freedMemory)
{
    {
        auto g = sstl::with_guard(*mu);
        lanes.remove_if([](auto &wl) {
            if (auto l = wl.lock()) {
                return false;
            }
            return true;
        });

        // NOTE: it's possible that the above remove_tf removes more than
        // one deleted lane, so the availableMemory here will not
        // correctly represent the actual amount.
        // But it's ok. Because eventually another call of laneRemoved
        // will correct that and serve queued requests.
        availableMemory += freedMemory;
        CHECK_LE(availableMemory, totalMemory);
    }

    // NOTE: may block or take long time
    mgr.processRequests();
}

GpuLane::GpuLane(LaneMgr::GpuControlBlock &gcb, size_t memoryLimit, int baseStreamIndex)
    : m_gcb(gcb)
    , m_availableMemory(memoryLimit)
    , m_baseStreamIndex(baseStreamIndex)
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

#if defined(SALUS_ENABLE_SIEXECUTOR)
    auto max_streams = 1;
#else
    // TODO: detect maximum streams in GPU
    auto max_streams = 128;
#endif

    tf::SessionOptions opt;
    m_dev = std::make_unique<SalusGPUDevice>(opt, name, allocated_bytes, dev_locality, m_gcb.id,
                                             GetShortDeviceDescription(m_gcb.id, desc), getGPUAllocator(),
                                             process_state->GetCPUAllocator(numa_node), m_gcb.cudaHostAlloc(),
                                             max_streams);
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

GpuLane::~GpuLane()
{
    // first release all resources in the lane
    m_dev.reset();

    // notify the control block about the removal
    m_gcb.laneRemoved(availableMemory());
}

} // namespace salus::oplib::tensorflow
