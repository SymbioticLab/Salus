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

#ifndef SALUS_OPLIB_TENSORFLOW_DEVICE_GPU_H
#define SALUS_OPLIB_TENSORFLOW_DEVICE_GPU_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "utils/objectpool.h"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace salus::oplib::tensorflow {

using NodeStreamMap = std::unordered_map<int, int>;

class PerTaskGPUDevice;
class SalusGPUDevice : public ISalusDevice, public tf::BaseGPUDevice
{
public:
    SalusGPUDevice(const tf::SessionOptions &options, const std::string &name, tf::Bytes memory_limit,
                   const tf::DeviceLocality &locality, int gpu_id, const std::string &physical_device_desc,
                   tf::Allocator *gpu_allocator, tf::Allocator *cpu_allocator, tf::Allocator *cuda_host_alloc, int max_streams = 128);

    ~SalusGPUDevice() override = default;

    tf::Allocator *GetAllocator(tf::AllocatorAttributes attr) override;

    Status Sync() override;

    bool RequiresRecordingAccessedTensors() const override;

    Status FillContextMap(const tf::Graph *graph, std::vector<tf::DeviceContext *> *device_context_map) override;

    void flushCacheFor(sstl::not_null<const tf::Graph *> graph) override;

    std::shared_ptr<PerTaskDevice> createPerTaskDevice(sstl::not_null<const tf::Graph *> graph,
                                                       std::unique_ptr<ResourceContext> &&rctx) override;

    std::unique_ptr<ShadowDevice> createSessionDevice(std::string newBaseName, std::string sessHandle) override;

    tf::Device &as_tfdevice() override
    {
        return *this;
    }

    const tf::Device &as_tfdevice() const override
    {
        return *this;
    }

private:
    /**
     * @brief Try to allocate streams. May return less than requested
     *
     * @param num
     * @return
     */
    std::vector<int> allocateStreams(size_t num);

    /**
     * @brief Free streams
     * @param streams
     */
    void freeStreams(std::vector<int> &&streams);

    /**
     * @brief Get the device context correspond to stream `num'
     * @param num
     * @return
     */
    sstl::not_null<tf::DeviceContext *> deviceContext(int num) const
    {
        DCHECK_LT(num, static_cast<int>(device_contexts_.size()));
        return device_contexts_[num];
    }

    friend class PerTaskGPUDevice;
    friend class SessionDevice;

    std::mutex m_muStream;
    std::vector<bool> m_streamUsed;
    tf::Allocator *m_cudaHostAlloc;
};

class SalusGPUDeviceFactory : public tf::BaseGPUDeviceFactory
{
private:
    tf::BaseGPUDevice *CreateGPUDevice(const tf::SessionOptions &options, const std::string &name,
                                       tf::Bytes memory_limit, const tf::DeviceLocality &locality, int gpu_id,
                                       const std::string &physical_device_desc, tf::Allocator *gpu_allocator,
                                       tf::Allocator *cpu_allocator) override;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_DEVICE_GPU_H
