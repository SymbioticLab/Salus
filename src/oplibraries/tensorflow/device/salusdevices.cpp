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

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "execution/engine/resourcecontext.h"
#include "oplibraries/tensorflow/device/cpu.h"
#include "oplibraries/tensorflow/device/gpu.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "utils/threadutils.h"

#include <mutex>

namespace salus::oplib::tensorflow {

namespace {

void registerSalusDeviceFactories()
{
    VLOG(1) << "Registering salus devices";
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

PerTaskDevice::PerTaskDevice(sstl::not_null<tf::Device *> other, std::unique_ptr<ResourceContext> &&rctx)
    : Device(other->env(), other->attributes())
    , m_base(other)
    , m_rctx(std::move(rctx))
{
    reinitialize();
}

void PerTaskDevice::reset(sstl::not_null<tf::Device *> other, std::unique_ptr<ResourceContext> &&rctx)
{
    m_base = other;
    m_rctx = std::move(rctx);
    m_wrappedAllocators.clear();
    reinitialize();

    if (!m_rctx) {
        LOG(ERROR) << "Resetting to an uninitialized resource context";
    }
}

void PerTaskDevice::reinitialize()
{
    set_tensorflow_cpu_worker_threads(
        const_cast<CpuWorkerThreads *>(m_base->tensorflow_cpu_worker_threads()));
    set_tensorflow_gpu_device_info(const_cast<GpuDeviceInfo *>(m_base->tensorflow_gpu_device_info()));
    set_eigen_cpu_device(const_cast<Eigen::ThreadPoolDevice *>(m_base->eigen_cpu_device()));
#ifdef TENSORFLOW_USE_SYCL
    set_eigen_sycl_device(const_cast<Eigen::SyclDevice *>(m_base->eigen_sycl_device()));
#endif
}

PerTaskDevice::~PerTaskDevice() = default;

bool PerTaskDevice::RequiresRecordingAccessedTensors() const
{
    return m_base->RequiresRecordingAccessedTensors();
}

tf::PerOpGpuDevice *PerTaskDevice::MakeGpuDevice()
{
    return m_base->MakeGpuDevice();
}

void PerTaskDevice::ReinitializeGpuDevice(tf::OpKernelContext *context, tf::PerOpGpuDevice *device,
                                          tf::DeviceContext *dc, tf::Allocator *allocator)
{
    m_base->ReinitializeGpuDevice(context, device, dc, allocator);
}

tf::Status PerTaskDevice::MakeTensorFromProto(const tf::TensorProto &tensor_proto,
                                              const tf::AllocatorAttributes alloc_attrs, tf::Tensor *tensor)
{
    return m_base->MakeTensorFromProto(tensor_proto, alloc_attrs, tensor);
}

void PerTaskDevice::Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context)
{
    m_base->Compute(op_kernel, context);
}

void PerTaskDevice::ComputeAsync(tf::AsyncOpKernel *op_kernel, tf::OpKernelContext *context,
                                 tf::AsyncOpKernel::DoneCallback done)
{
    m_base->ComputeAsync(op_kernel, context, std::move(done));
}

void PerTaskDevice::ConsumeListOfAccessedTensors(tf::DeviceContext *context,
                                                 const tf::TensorReferenceVector &tensors)
{
    m_base->ConsumeListOfAccessedTensors(context, tensors);
}

tf::Status PerTaskDevice::Sync()
{
    return m_base->Sync();
}

tf::Status PerTaskDevice::MaybeRewriteGraph(std::unique_ptr<tf::Graph> *graph)
{
    return m_base->MaybeRewriteGraph(graph);
}

tf::Status PerTaskDevice::FillContextMap(const tf::Graph *graph, tf::DeviceContextMap *device_context_map)
{
    return m_base->FillContextMap(graph, device_context_map);
}

tf::ResourceMgr *PerTaskDevice::resource_manager()
{
    return m_base->resource_manager();
}

tf::Allocator *PerTaskDevice::GetAllocator(tf::AllocatorAttributes attr)
{
    auto a = m_base->GetAllocator(attr);
    return wrapAllocator(a, attr);
}

tf::Allocator *PerTaskDevice::GetStepAllocator(tf::AllocatorAttributes attr,
                                               tf::ResourceMgr *step_resource_manager)
{
    auto a = m_base->GetStepAllocator(attr, step_resource_manager);
    return wrapAllocator(a, attr);
}

tf::Allocator *PerTaskDevice::wrapAllocator(tf::Allocator *alloc, const tf::AllocatorAttributes &attr)
{
    DCHECK(m_rctx);

    AA key{alloc, attr};

    auto g = sstl::with_guard(m_mu);
    auto it = m_wrappedAllocators.find(key);
    if (it != m_wrappedAllocators.end()) {
        return it->second.get();
    }

    sstl::ScopedUnref<PerOpAllocator> a;
    if (attr.on_host()) {
        DCHECK(alloc->Name() != "GPU_0_bfc");
        DeviceSpec cpuSpec{DeviceType::CPU, 0};
        auto rctx = std::make_unique<ResourceContext>(*m_rctx, cpuSpec);
        a = sstl::make_scoped_unref<PerOpAllocator>(std::move(rctx), alloc);
    } else {
        a = sstl::make_scoped_unref<PerOpAllocator>(m_rctx, alloc);
    }
    auto pa = a.get();
    m_wrappedAllocators.emplace(key, std::move(a));
    return pa;
}

Resources PerTaskDevice::failedResourceRequest() const
{
    Resources res;
    auto g = sstl::with_guard(m_mu);
    for (auto &p : m_wrappedAllocators) {
        auto alloc = p.second.get();
        res[{ResourceType::MEMORY, alloc->resourceContext().spec()}] += alloc->lastFailedAllocSize();
    }
    return res;
}

Resources PerTaskDevice::peakResourceUsage() const
{
    Resources res;
    auto g = sstl::with_guard(m_mu);
    for (const auto &[aa, alloc] : m_wrappedAllocators) {
        UNUSED(aa);
        res[{ResourceType::MEMORY, alloc->resourceContext().spec()}] += alloc->peakAllocSize();
    }
    return res;
}

} // namespace salus::oplib::tensorflow
