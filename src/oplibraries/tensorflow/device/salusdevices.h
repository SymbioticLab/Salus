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

#ifndef SALUS_OPLIB_TENSORFLOW_SALUSDEVICES
#define SALUS_OPLIB_TENSORFLOW_SALUSDEVICES

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "resources/resources.h"

#include <functional>
#include <memory>
#include <mutex>

namespace tensorflow {
class Device;
} // namespace tensorflow

namespace salus::oplib::tensorflow {

class ResourceContext;
class PerTaskDevice;
class PerOpAllocator;
class ShadowDevice;
/**
 * @brief We use an extension to tensorflow devices.
 *
 * All TF devices should implement this interface.
 */
class ISalusDevice
{
public:
    virtual ~ISalusDevice() = default;

    virtual void flushCacheFor(sstl::not_null<const tf::Graph *> graph) = 0;

    virtual std::shared_ptr<PerTaskDevice> createPerTaskDevice(sstl::not_null<const tf::Graph *> g,
                                                               std::unique_ptr<ResourceContext> &&rctx) = 0;

    virtual std::unique_ptr<ShadowDevice> createSessionDevice(std::string newBaseName, std::string sessHandle) = 0;

    /**
     * @brief Get the tf::Device instance
     * @return
     */
    virtual tf::Device &as_tfdevice() = 0;
    virtual const tf::Device &as_tfdevice() const = 0;

    /**
     * @brief Safely cast a tf::Device to ISalusDeivce, w/o dynamic_cast and RTTI.
     *
     * It does this by first downcasting to concrete device type and then upcast,
     * based on the device name and type.
     *
     * @param device
     * @return
     */
    static ISalusDevice *safe_cast(tf::Device *device);
    static ISalusDevice &safe_cast(tf::Device &device);
};

void maybeRegisterSalusDeviceFactories();

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_SALUSDEVICES
