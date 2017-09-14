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

#include "peropallocdevice.h"

#include "utils/threadutils.h"
#include "oplibraries/tensorflow/v2/tfallocator.h"

PerOpAllocDevice::PerOpAllocDevice(tf::Device *other)
    : Device(other->env(), other->attributes(), nullptr)
    , m_wrapped(other)
{
    assert(m_wrapped);

    set_tensorflow_cpu_worker_threads(const_cast<CpuWorkerThreads*>(other->tensorflow_cpu_worker_threads()));
    set_tensorflow_gpu_device_info(const_cast<GpuDeviceInfo*>(other->tensorflow_gpu_device_info()));
    set_eigen_cpu_device(const_cast<Eigen::ThreadPoolDevice*>(other->eigen_cpu_device()));
#ifdef TENSORFLOW_USE_SYCL
    set_eigen_sycl_device(const_cast<Eigen::SyclDevice*>(other->eigen_sycl_device()));
#endif
}

PerOpAllocDevice::~PerOpAllocDevice() = default;

void PerOpAllocDevice::setResourceContext(const ResourceContext &rctx)
{
    m_rctx = rctx;
}

bool PerOpAllocDevice::RequiresRecordingAccessedTensors() const
{
    return m_wrapped->RequiresRecordingAccessedTensors();
}

tf::PerOpGpuDevice* PerOpAllocDevice::MakeGpuDevice()
{
    return m_wrapped->MakeGpuDevice();
}

void PerOpAllocDevice::ReinitializeGpuDevice(tf::OpKernelContext* context,
                            tf::PerOpGpuDevice* device,
                            tf::DeviceContext* dc,
                            tf::Allocator* allocator)
{
    m_wrapped->ReinitializeGpuDevice(context, device, dc, allocator);
}

tf::Status PerOpAllocDevice::MakeTensorFromProto(const tf::TensorProto& tensor_proto,
                                                 const tf::AllocatorAttributes alloc_attrs,
                                                 tf::Tensor* tensor)
{
    return m_wrapped->MakeTensorFromProto(tensor_proto, alloc_attrs, tensor);
}

void PerOpAllocDevice::Compute(tf::OpKernel* op_kernel, tf::OpKernelContext* context)
{
    m_wrapped->Compute(op_kernel, context);
}

void PerOpAllocDevice::ComputeAsync(tf::AsyncOpKernel* op_kernel, tf::OpKernelContext* context,
                                    tf::AsyncOpKernel::DoneCallback done)
{
    m_wrapped->ComputeAsync(op_kernel, context, std::move(done));
}

void PerOpAllocDevice::ConsumeListOfAccessedTensors(tf::DeviceContext* context,
                                                    const tf::TensorReferenceVector& tensors)
{
    m_wrapped->ConsumeListOfAccessedTensors(context, tensors);
}

tf::Status PerOpAllocDevice::Sync()
{
    return m_wrapped->Sync();
}

tf::Status PerOpAllocDevice::MaybeRewriteGraph(const tf::FunctionDefLibrary& library,
                                               std::unique_ptr<tf::Graph>* graph)
{
    return m_wrapped->MaybeRewriteGraph(library, graph);
}

tf::Status PerOpAllocDevice::FillContextMap(const tf::Graph* graph, tf::DeviceContextMap* device_context_map)
{
    return m_wrapped->FillContextMap(graph, device_context_map);
}

tf::Allocator* PerOpAllocDevice::GetAllocator(tf::AllocatorAttributes attr)
{
    auto a = m_wrapped->GetAllocator(attr);
    return wrapAllocator(a);
}

tf::Allocator* PerOpAllocDevice::GetStepAllocator(tf::AllocatorAttributes attr,
                                tf::ResourceMgr* step_resource_manager)
{
    auto a = m_wrapped->GetStepAllocator(attr, step_resource_manager);
    return wrapAllocator(a);
}

tf::Allocator *PerOpAllocDevice::wrapAllocator(tf::Allocator *alloc)
{
    assert(m_rctx.resMon);

    utils::Guard g(m_mu);
    auto it = m_wrappedAllocators.find(alloc);
    if (it != m_wrappedAllocators.end()) {
        return it->second.get();
    }

    auto a = utils::make_scoped_unref<PerOpAllocator>(m_rctx.ticket, m_rctx.spec, *m_rctx.resMon, alloc);
    auto pa = a.get();
    m_wrappedAllocators.emplace(alloc, std::move(a));
    return pa;
}
