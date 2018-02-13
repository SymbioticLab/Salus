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

#ifndef SALUS_OPLIB_TENSORFLOW_DEVICECONTEXTWITHDEVICE_H
#define SALUS_OPLIB_TENSORFLOW_DEVICECONTEXTWITHDEVICE_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "utils/pointerutils.h"
#include <memory>

namespace salus::oplib::tensorflow {

/**
 * @brief A thin wrapper of tf::DeviceContext that also provides a device pointer
 */
class DeviceContextWithDevice : public tf::DeviceContext
{
public:
    using ReffedDeviceContext = sstl::ScopedUnref<DeviceContext>;

    /**
     * @param actual Takes one ref on actual
     */
    DeviceContextWithDevice(std::shared_ptr<tf::Device> dev, ReffedDeviceContext &&actual);

    ~DeviceContextWithDevice() override;

    perftools::gputools::Stream *stream() const override;

    void MaintainLifetimeOnStream(const tf::Tensor *t,
                                  perftools::gputools::Stream *stream) const override;

    void CopyCPUTensorToDevice(const tf::Tensor *cpu_tensor, tf::Device *device,
                               tf::Tensor *device_tensor,
                               tf::StatusCallback done) const override;

    void CopyDeviceTensorToCPU(const tf::Tensor *device_tensor,
                               tf::StringPiece tensor_name, tf::Device *device,
                               tf::Tensor *cpu_tensor, tf::StatusCallback done) override;

    tf::Device *device() const
    {
        return m_device.get();
    }

    tf::DeviceContext *wrapped() const
    {
        return m_actualCtx.get();
    }

private:
    std::shared_ptr<tf::Device> m_device;
    ReffedDeviceContext m_actualCtx;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_DEVICECONTEXTWITHDEVICE_H
