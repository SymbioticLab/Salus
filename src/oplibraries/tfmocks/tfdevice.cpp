/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017  Aetf <aetf@unlimitedcodeworks.xyz>
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
 * 
 */

#include "tfdevice.h"

#include "tfallocator.h"

#include "platform/logging.h"
#include "utils/macros.h"

TFDevice::TFDevice(const tensorflow::SessionOptions &options)
    : LocalDevice(options,
                  Device::BuildDeviceAttributes("/device:CPU:0", tensorflow::DEVICE_CPU,
                                                tensorflow::Bytes(256 << 20), tensorflow::DeviceLocality()),
                  nullptr)
    , m_allocator(new TFAllocator)
{ }

TFDevice::~TFDevice() = default;

void TFDevice::Compute(tensorflow::OpKernel* op_kernel, tensorflow::OpKernelContext* context)
{
    UNUSED(op_kernel);
    UNUSED(context);
    WARN("Empty compute");
}

tensorflow::Allocator* TFDevice::GetAllocator(tensorflow::AllocatorAttributes attr)
{
    UNUSED(attr);
    return m_allocator.get();
}

tensorflow::Status TFDevice::MakeTensorFromProto(const tensorflow::TensorProto& tensor_proto,
                            const tensorflow::AllocatorAttributes alloc_attrs,
                            tensorflow::Tensor* tensor)
{
    INFO("TFDevice::MakeTensorFromProto got tensor_proto {}", tensor_proto.DebugString());
    INFO("TFDevice::MakeTensorFromProto got alloc_attrs {}", alloc_attrs);

    if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= tensorflow::DataType_MAX) {
        tensorflow::Tensor parsed(tensor_proto.dtype());
        if (parsed.FromProto(m_allocator.get(), tensor_proto)) {
            if (parsed.shape().num_elements() == 0) {
                // Unallocated tensor, we have to allocate a value anyway.
                tensorflow::Tensor t(m_allocator.get(), tensor_proto.dtype(), parsed.shape());
                *tensor = t;
            } else {
                *tensor = parsed;
            }
            return tensorflow::Status::OK();
        }
    }
    ERR("Cannot parse tensor from proto: {}", tensor_proto.DebugString());
    return tensorflow::errors::InvalidArgument("Cannot parse tensor from proto: ",
                                               tensor_proto.DebugString());
}
