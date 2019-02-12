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
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "execution/engine/resourcecontext.h"
#include "oplibraries/tensorflow/device/cpu.h"
#include "oplibraries/tensorflow/device/gpu/gpu.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "utils/threadutils.h"

#include <mutex>

namespace salus::oplib::tensorflow {

namespace {

void registerSalusDeviceFactories()
{
    VLOG(2) << "Registering salus devices";
    REGISTER_LOCAL_DEVICE_FACTORY("CPU", SalusCPUDeviceFactory, 999);
    REGISTER_LOCAL_DEVICE_FACTORY("GPU", SalusGPUDeviceFactory, 999);
}
} // namespace

void maybeRegisterSalusDeviceFactories()
{
    static std::once_flag flag;
    std::call_once(flag, registerSalusDeviceFactories);
}

ISalusDevice *ISalusDevice::safe_cast(tf::Device *device)
{
    if (!device) {
        return nullptr;
    }

    if (device->device_type() == tf::DEVICE_CPU) {
        return static_cast<SalusCPUDevice *>(device);
    } else if (device->device_type() == tf::DEVICE_GPU) {
        return static_cast<SalusGPUDevice *>(device);
    } else {
        LOG(WARNING) << "ISalusDevice::safe_cast got unknown device type: " << device->device_type();
        return nullptr;
    }
}

ISalusDevice &ISalusDevice::safe_cast(tf::Device &device)
{
    auto sdev = safe_cast(&device);
    if (!sdev) {
        throw TFException(tf::errors::InvalidArgument("device is not an ISalusDevice: ", device.name()));
    }
    return *sdev;
}

} // namespace salus::oplib::tensorflow
