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

#ifndef SALUS_OPLIB_TENSORFLOW_DEVICE_CPU_H
#define SALUS_OPLIB_TENSORFLOW_DEVICE_CPU_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "utils/pointerutils.h"
#include "utils/objectpool.h"

namespace salus::oplib::tensorflow {

class PerTaskCPUDevice;
class SalusCPUDevice : public ISalusDevice, public tf::LocalDevice
{
public:
    SalusCPUDevice(const tf::SessionOptions &options, const std::string &name, tf::Bytes memory_limit,
                   const tf::DeviceLocality &locality, tf::Allocator *allocator, tf::Allocator *cudaAlloc = nullptr);

    ~SalusCPUDevice() override = default;

    tf::Allocator *GetAllocator(tf::AllocatorAttributes attr) override;

    Status Sync() override
    {
        return Status::OK();
    }

    void Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context) override
    {
        op_kernel->Compute(context);
    }

    Status MakeTensorFromProto(const tf::TensorProto &tensor_proto, const tf::AllocatorAttributes alloc_attrs,
                               tf::Tensor *tensor) override;

    void flushCacheFor(sstl::not_null<const tf::Graph *>) override
    {
    }

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
    sstl::not_null<tf::Allocator *> m_allocator; // not owned
    tf::Allocator * m_cudaAlloc; // not owned
};

class SalusCPUDeviceFactory : public tf::DeviceFactory
{
public:
    Status CreateDevices(const tf::SessionOptions &options, const std::string &name_prefix,
                         std::vector<tf::Device *> *devices) override;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_DEVICE_CPU_H
