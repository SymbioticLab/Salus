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

#include "devicecontextwithdevice.h"

namespace tf = ::tensorflow;

namespace salus::oplib::tensorflow {

DeviceContextWithDevice::DeviceContextWithDevice(std::shared_ptr<tf::Device> dev, ReffedDeviceContext &&actual)
    : m_device(std::move(dev))
    , m_actualCtx(std::move(actual))
{
    DCHECK(m_device);
}

DeviceContextWithDevice::~DeviceContextWithDevice() = default;

perftools::gputools::Stream *DeviceContextWithDevice::stream() const
{
    if (m_actualCtx) {
        return m_actualCtx->stream();
    } else {
        return DeviceContext::stream();
    }
}

void DeviceContextWithDevice::MaintainLifetimeOnStream(const tf::Tensor *t,
                                                       perftools::gputools::Stream *stream) const
{
    if (m_actualCtx) {
        return m_actualCtx->MaintainLifetimeOnStream(t, stream);
    } else {
        return DeviceContext::MaintainLifetimeOnStream(t, stream);
    }
}

void DeviceContextWithDevice::CopyCPUTensorToDevice(const tf::Tensor *cpu_tensor, tf::Device *device,
                                                    tf::Tensor *device_tensor, tf::StatusCallback done) const
{
    if (m_actualCtx) {
        return m_actualCtx->CopyCPUTensorToDevice(cpu_tensor, device, device_tensor, done);
    } else {
        return DeviceContext::CopyCPUTensorToDevice(cpu_tensor, device, device_tensor, done);
    }
}

void DeviceContextWithDevice::CopyDeviceTensorToCPU(const tf::Tensor *device_tensor,
                                                    tf::StringPiece tensor_name, tf::Device *device,
                                                    tf::Tensor *cpu_tensor, tf::StatusCallback done)
{
    if (m_actualCtx) {
        return m_actualCtx->CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor, done);
    } else {
        return DeviceContext::CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor, done);
    }
}

} // namespace salus::oplib::tensorflow
