/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Peifeng Yu <peifeng@umich.edu>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
