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

#include "platform/logging.h"

TFDevice::TFDevice()
: Device(nullptr,
         Device::BuildDeviceAttributes("/device:CPU:0", tensorflow::DEVICE_CPU,
                                       tensorflow::Bytes(256 << 20), tensorflow::DeviceLocality()),
         nullptr)
{
    
}

TFDevice::~TFDevice() = default;

void TFDevice::Compute(tensorflow::OpKernel* op_kernel, tensorflow::OpKernelContext* context)
{
    WARN("Empty compute");
}

tensorflow::Allocator* TFDevice::GetAllocator(tensorflow::AllocatorAttributes attr)
{
    ERR("Not implemented: TFDEvice::GetAllocator");
    return nullptr;
}

tensorflow::Status TFDevice::MakeTensorFromProto(const tensorflow::TensorProto& tensor_proto,
                            const tensorflow::AllocatorAttributes alloc_attrs,
                            tensorflow::Tensor* tensor)
{
    return tensorflow::Status::OK();
}
