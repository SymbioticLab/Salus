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

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "oplibraries/tensorflow/device/gpu/gpu.h"

#include "execution/engine/resourcecontext.h"
#include "oplibraries/tensorflow/device/gpu/sessiondevice.h"
#include "utils/threadutils.h"

#include <thread>
#include <utility>

namespace salus::oplib::tensorflow {

SalusGPUDevice::SalusGPUDevice(const tf::SessionOptions &options, const std::string &name, tf::Bytes memory_limit,
                               const tf::DeviceLocality &locality, int gpu_id, const std::string &physical_device_desc,
                               tf::Allocator *gpu_allocator, tf::Allocator *cpu_allocator, tf::Allocator *cuda_host_alloc, int max_streams)
    : BaseGPUDevice(options, name, memory_limit, locality, gpu_id, physical_device_desc, gpu_allocator, cpu_allocator,
                    false /* sync every op */, max_streams)
    , m_streamUsed(static_cast<size_t>(max_streams), false)
    , m_cudaHostAlloc(cuda_host_alloc)
{
}

tf::Allocator *SalusGPUDevice::GetAllocator(tf::AllocatorAttributes attr)
{
    if (attr.on_host()) {
        if (attr.gpu_compatible()) {
            if (m_cudaHostAlloc) {
                return m_cudaHostAlloc;
            }
            return tf::ProcessState::singleton()->GetCUDAHostAllocator(0);
        } else {
            return cpu_allocator_;
        }
    }
    return gpu_allocator_;
}

Status SalusGPUDevice::Sync()
{
    return BaseGPUDevice::Sync();
    /*
    CHECK_EQ(streams_.size(), 1);

    const constexpr int PerGroupStreams = 1;
    sstl::semaphore sa;
    for (auto sg : streams_) {
        tensorflow_gpu_device_info()->event_mgr->ThenExecute(sg->compute, [&sa](){
            sa.notify();
        });
    }
    sa.wait(static_cast<uint32_t>(streams_.size() * PerGroupStreams));
    */

    /*
    std::vector<std::unique_ptr<tfgpu::Event>> events(streams_.size() * PerGroupStreams);
    for (auto &evt : events) {
        evt = std::make_unique<tfgpu::Event>(executor_);
        if (!evt->Init()) {
            return tf::errors::Internal("Failed to init gpu event when doing sync");
        }
    }

    auto it = events.begin();
    for (auto sg : streams_) {
        DCHECK_NE(it, events.end());
        sg->compute->ThenRecordEvent(it->get());
        ++it;
//        DCHECK_NE(it, events.end());
//        sg->host_to_device->ThenRecordEvent(it->get());
//        ++it;
//        DCHECK_NE(it, events.end());
//        sg->device_to_host->ThenRecordEvent(it->get());
//        ++it;
//        DCHECK_NE(it, events.end());
//        sg->device_to_device->ThenRecordEvent(it->get());
//        ++it;
    }

    // Start polling for the record of all events,
    // we will block to wait anyway, so polling is acceptable.
    // Just don't poll at too high freq.
    auto remaining = events.size();
    while (remaining) {
        for (auto &evt : events) {
            if (!evt) {
                continue;
            }
            auto es = evt->PollForStatus();
            switch (es) {
            case tfgpu::Event::Status::kUnknown:
            case tfgpu::Event::Status::kError:
                return tf::errors::Internal("GPU returned an error while sync");
            case tfgpu::Event::Status::kComplete:
                --remaining;
                evt.reset();
                break;
            case tfgpu::Event::Status::kPending:
                continue;
            }
        }

        if (remaining) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(10us);
        }
    }
    return Status::OK();
    */
}

bool SalusGPUDevice::RequiresRecordingAccessedTensors() const
{
    return BaseGPUDevice::RequiresRecordingAccessedTensors();
}

void SalusGPUDevice::flushCacheFor(sstl::not_null<const tf::Graph *>)
{
    //    VLOG(3) << "SalusGPUDevice::flushCacheFor(" << as_hex(graph) << ") on " << name();
    //    auto g = sstl::with_guard(m_muCache);
    //    m_streamAssignCache.erase(graph);
}

std::shared_ptr<PerTaskDevice> SalusGPUDevice::createPerTaskDevice(sstl::not_null<const tf::Graph *> graph,
                                                                   std::unique_ptr<ResourceContext> &&rctx)
{
    UNUSED(graph);
    UNUSED(rctx);
    return nullptr;
}

std::unique_ptr<ShadowDevice> SalusGPUDevice::createSessionDevice(std::string newBaseName, std::string sessHandle)
{
    static int NextStreamBase = 0;

    auto streamBase = NextStreamBase;
    NextStreamBase = (NextStreamBase + 1) % max_streams_;

    GpuDeviceInfo newInfo{*tensorflow_gpu_device_info()};
    newInfo.default_context = device_contexts_[streamBase];
    newInfo.stream = streams_[streamBase]->compute;

    std::vector<SessionDevice::StreamAndContext> scs{ {streams_[streamBase], device_contexts_[streamBase]} };

    auto d = std::make_unique<SessionDevice>(this, std::move(newBaseName), std::move(sessHandle),
                                             newInfo, std::move(scs));
    return d;
}

std::vector<int> SalusGPUDevice::allocateStreams(size_t num)
{
    if (num == 0) {
        return {};
    }

    auto g = sstl::with_guard(m_muStream);
    std::vector<int> res;
    for (int i = 0; i != max_streams_; ++i) {
        if (!m_streamUsed[i]) {
            res.emplace_back(i);
            m_streamUsed[i] = true;
        }

        if (res.size() == num) {
            break;
        }
    }
    return res;
}

void SalusGPUDevice::freeStreams(std::vector<int> &&streams)
{
    if (streams.empty()) {
        return;
    }

    auto g = sstl::with_guard(m_muStream);
    for (auto i : streams) {
        m_streamUsed[i] = false;
    }
    streams.clear();
}

Status SalusGPUDevice::FillContextMap(const tf::Graph *graph, std::vector<tf::DeviceContext *> *device_context_map)
{
    return BaseGPUDevice::FillContextMap(graph, device_context_map);
}

tf::BaseGPUDevice *SalusGPUDeviceFactory::CreateGPUDevice(const tf::SessionOptions &options, const std::string &name,
                                                          tf::Bytes memory_limit, const tf::DeviceLocality &locality,
                                                          int gpu_id, const std::string &physical_device_desc,
                                                          tf::Allocator *gpu_allocator, tf::Allocator *cpu_allocator)
{
    auto max_streams = 1;

    auto dev = std::make_unique<SalusGPUDevice>(options, name, memory_limit, locality, gpu_id, physical_device_desc,
                                                gpu_allocator, cpu_allocator, nullptr, max_streams);
    return dev.release();
}

} // namespace salus::oplib::tensorflow
