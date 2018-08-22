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
