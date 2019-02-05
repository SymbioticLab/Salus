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

#ifndef SALUS_OPLIB_TENSORFLOW_SESSIONDEVICE_H
#define SALUS_OPLIB_TENSORFLOW_SESSIONDEVICE_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "oplibraries/tensorflow/device/shadowdevices.h"
#include "oplibraries/tensorflow/device/gpu/gpu.h"

#include <vector>
#include <utility>

namespace salus::oplib::tensorflow {

/**
 * @brief This device is only for GPU
 */
class SessionDevice : public ShadowDevice
{
public:
    using StreamAndContext = std::pair<sstl::not_null<SalusGPUDevice::StreamGroup*>,
        sstl::not_null<tf::GPUDeviceContext*>>;
    explicit SessionDevice(sstl::not_null<tf::Device *> base, const std::string &newBaseName, std::string sessHandle,
                           GpuDeviceInfo newInfo, std::vector<StreamAndContext> streams);

    tf::Status Sync() override;
    tf::Status FillContextMap(const tf::Graph *graph, tf::DeviceContextMap *device_context_map) override;

private:
    sstl::ScopedUnref<ForwardingAllocator> createWrappedAllocator(tf::Allocator *alloc,
                                                                  const tf::AllocatorAttributes &attrs);

    const std::string m_sessHandle;
    GpuDeviceInfo m_gpuDeviceInfo;
    std::vector<StreamAndContext> m_streams;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_SESSIONDEVICE_H
