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

#ifndef SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_DEVICECONTEXTWITHDEVICE_H
#define SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_DEVICECONTEXTWITHDEVICE_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "utils/pointerutils.h"
#include <memory>

namespace symbiotic::salus::oplib::tensorflow {

/**
 * @brief A thin wrapper of ::tensorflow::DeviceContext that also provides a device pointer
 */
class DeviceContextWithDevice : public ::tensorflow::DeviceContext
{
public:
    using ReffedDeviceContext = utils::ScopedUnref<DeviceContext>;

    /**
     * @param actual Takes one ref on actual
     */
    DeviceContextWithDevice(std::shared_ptr<::tensorflow::Device> dev, ReffedDeviceContext &&actual);

    ~DeviceContextWithDevice() override;

    perftools::gputools::Stream *stream() const override;

    void MaintainLifetimeOnStream(const ::tensorflow::Tensor *t,
                                  perftools::gputools::Stream *stream) const override;

    void CopyCPUTensorToDevice(const ::tensorflow::Tensor *cpu_tensor, ::tensorflow::Device *device,
                               ::tensorflow::Tensor *device_tensor,
                               ::tensorflow::StatusCallback done) const override;

    void CopyDeviceTensorToCPU(const ::tensorflow::Tensor *device_tensor,
                               ::tensorflow::StringPiece tensor_name, ::tensorflow::Device *device,
                               ::tensorflow::Tensor *cpu_tensor, ::tensorflow::StatusCallback done) override;

    ::tensorflow::Device *device() const
    {
        return m_device.get();
    }

    ::tensorflow::DeviceContext *wrapped() const
    {
        return m_actualCtx.get();
    }

private:
    std::shared_ptr<::tensorflow::Device> m_device;
    ReffedDeviceContext m_actualCtx;
};

} // namespace symbiotic::salus::oplib::tensorflow

#endif // SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_DEVICECONTEXTWITHDEVICE_H
