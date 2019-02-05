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

#include "oplibraries/tensorflow/device/cpu.h"

#include "execution/executionengine.h"
#include "oplibraries/tensorflow/device/shadowdevices.h"
#include "oplibraries/tensorflow/device/sessionallocator.h"
#include "utils/objectpool.h"
#include "utils/threadutils.h"

namespace salus::oplib::tensorflow {

SalusCPUDevice::SalusCPUDevice(const tf::SessionOptions &options, const std::string &name, tf::Bytes memory_limit,
                               const tf::DeviceLocality &locality, tf::Allocator *allocator, tf::Allocator *cudaAlloc)
    : LocalDevice(options, tf::Device::BuildDeviceAttributes(name, tf::DEVICE_CPU, memory_limit, locality))
    , m_allocator(allocator)
    , m_cudaAlloc(cudaAlloc)
{
}

tf::Allocator *SalusCPUDevice::GetAllocator(tf::AllocatorAttributes attr)
{
    if (attr.gpu_compatible()) {
        if (m_cudaAlloc) {
            return m_cudaAlloc;
        }
        return tf::ProcessState::singleton()->GetCUDAHostAllocator(0);
    }
    return m_allocator;
}

Status SalusCPUDevice::MakeTensorFromProto(const tf::TensorProto &tensor_proto,
                                           const tf::AllocatorAttributes alloc_attrs, tf::Tensor *tensor)
{
    UNUSED(alloc_attrs);
    if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= tf::DataType_MAX) {
        tf::Tensor parsed(tensor_proto.dtype());
        if (parsed.FromProto(m_allocator, tensor_proto)) {
            *tensor = std::move(parsed);
            return Status::OK();
        }
    }
    return tf::errors::InvalidArgument("Cannot parse tensor from proto: ", tf::ProtoDebugString(tensor_proto));
}

std::shared_ptr<PerTaskDevice> SalusCPUDevice::createPerTaskDevice(sstl::not_null<const tf::Graph *> graph,
                                                                   std::unique_ptr<ResourceContext> &&rctx)
{
    UNUSED(graph);
    UNUSED(rctx);
    return nullptr;
}

std::unique_ptr<ShadowDevice> SalusCPUDevice::createSessionDevice(std::string newBaseName, std::string sessHandle)
{
    return ShadowDevice::NewShadowDevice(newBaseName, this, true, false, [sessHandle = std::move(sessHandle)](auto alloc, auto){
        return sstl::make_scoped_unref<SessionAllocator>(sessHandle, alloc);
    });
}

Status SalusCPUDeviceFactory::CreateDevices(const tf::SessionOptions &options, const std::string &name_prefix,
                                            std::vector<tf::Device *> *devices)
{
    // TODO(zhifengc/tucker): Figure out the number of available CPUs and/or NUMA configuration.
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
        n = iter->second;
    }
    for (int i = 0; i < n; i++) {
        auto name = tf::strings::StrCat(name_prefix, "/cpu:", i);
        // use tf::cpu_allocator to select from cpu allocatory registary
        auto dev = new SalusCPUDevice(options, name, tf::Bytes(256 << 20), {}, tf::cpu_allocator());
        VLOG(3) << "Creating SalusCPUDevice " << as_hex(dev) << " which is a tf::Device "
                << as_hex(static_cast<tf::Device *>(dev)) << " and also a ISalusDevice "
                << as_hex(static_cast<ISalusDevice *>(dev));
        devices->push_back(dev);
    }

    return Status::OK();
}

} // namespace salus::oplib::tensorflow
