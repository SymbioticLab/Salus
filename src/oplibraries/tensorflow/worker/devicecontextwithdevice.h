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
