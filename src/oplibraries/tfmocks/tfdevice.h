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

#ifndef TFDEVICE_H
#define TFDEVICE_H

#include <tensorflow/core/common_runtime/device.h>

#include <memory>

class TFAllocator;
/**
 * @todo write docs
 */
class TFDevice : public tensorflow::Device
{
public:
    TFDevice();
    ~TFDevice() override;

    void Compute(tensorflow::OpKernel* op_kernel, tensorflow::OpKernelContext* context) override;
    tensorflow::Allocator* GetAllocator(tensorflow::AllocatorAttributes attr) override;
    tensorflow::Status MakeTensorFromProto(const tensorflow::TensorProto& tensor_proto,
                               const tensorflow::AllocatorAttributes alloc_attrs,
                               tensorflow::Tensor* tensor) override;

    tensorflow::Status Sync() override { return tensorflow::Status::OK(); }

private:
    std::unique_ptr<TFAllocator> m_allocator;
};

#endif // TFDEVICE_H
