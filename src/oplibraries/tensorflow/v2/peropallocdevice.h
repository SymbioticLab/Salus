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
 */

#ifndef PEROPALLOCDEVICE_H
#define PEROPALLOCDEVICE_H

#include "utils/pointerutils.h"

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include <mutex>

struct ResourceContext;
class PerOpAllocator;
class PerOpAllocDevice : public tf::Device
{
public:
    explicit PerOpAllocDevice(tf::Device *other);
    ~PerOpAllocDevice() override;

    void setResourceContext(const std::shared_ptr<ResourceContext> &rctx);
    const ResourceContext &resourceContext() const { return *m_rctx; }

    tf::Device *underlayingDevice() const {
        return m_wrapped;
    }

    tf::Allocator* GetAllocator(tf::AllocatorAttributes attr) override;

    tf::Allocator* GetStepAllocator(tf::AllocatorAttributes attr,
                                    tf::ResourceMgr* step_resource_manager) override;

    // Forwarding of DeviceBase methods
    bool RequiresRecordingAccessedTensors() const override;

    tf::PerOpGpuDevice* MakeGpuDevice() override;
    void ReinitializeGpuDevice(tf::OpKernelContext* context, tf::PerOpGpuDevice* device,
                               tf::DeviceContext* dc, tf::Allocator* allocator) override;

    tf::Status MakeTensorFromProto(const tf::TensorProto& tensor_proto,
                                   const tf::AllocatorAttributes alloc_attrs,
                                   tf::Tensor* tensor) override;

    // Forwarding of Device methods
    void Compute(tf::OpKernel* op_kernel, tf::OpKernelContext* context) override;
    void ComputeAsync(tf::AsyncOpKernel* op_kernel, tf::OpKernelContext* context,
                      tf::AsyncOpKernel::DoneCallback done) override;
    void ConsumeListOfAccessedTensors(tf::DeviceContext* context, const tf::TensorReferenceVector& tensors) override;
    tf::Status Sync() override;
    tf::Status MaybeRewriteGraph(const tf::FunctionDefLibrary& library, std::unique_ptr<tf::Graph>* graph) override;
    tf::Status FillContextMap(const tf::Graph* graph, tf::DeviceContextMap* device_context_map) override;

private:

    tf::Allocator *wrapAllocator(tf::Allocator *alloc);

    tf::Device *m_wrapped;
    std::shared_ptr<ResourceContext> m_rctx;

    std::mutex m_mu;
    std::unordered_map<tf::Allocator*, utils::ScopedUnref<PerOpAllocator>> m_wrappedAllocators;
};

#endif // PEROPALLOCDEVICE_H
