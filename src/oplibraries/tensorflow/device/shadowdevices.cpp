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

#include "oplibraries/tensorflow/device/shadowdevices.h"

#include "utils/threadutils.h"

namespace salus::oplib::tensorflow {

ForwardingAllocator::ForwardingAllocator(sstl::not_null<tf::Allocator *> actual,
                                         const std::string &namePrefix)
    : m_base(actual)
    , m_prefix(namePrefix)
{
}

ForwardingAllocator::~ForwardingAllocator() = default;

std::string ForwardingAllocator::Name()
{
    return tf::strings::StrCat(m_prefix, m_base->Name());
}

bool ForwardingAllocator::preAllocation(size_t, size_t, const tf::AllocationAttributes &)
{
    return true;
}

void *ForwardingAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
{
    if (!preAllocation(alignment, num_bytes, {})) {
        return nullptr;
    }

    auto ptr = m_base->AllocateRaw(alignment, num_bytes);
    recordSize(ptr, num_bytes);
    postAllocation(ptr, alignment, num_bytes, {});
    return ptr;
}

void *ForwardingAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
                                       const tf::AllocationAttributes &allocation_attr)
{
    if (!preAllocation(alignment, num_bytes, allocation_attr)) {
        return nullptr;
    }

    auto ptr = m_base->AllocateRaw(alignment, num_bytes, allocation_attr);
    recordSize(ptr, num_bytes);
    postAllocation(ptr, alignment, num_bytes, allocation_attr);
    return ptr;
}

void ForwardingAllocator::postAllocation(void *ptr, size_t alignment, size_t num_bytes,
                                         const tf::AllocationAttributes &)
{
    if (ptr) {
        LogAlloc() << "event: alloc "
                   << as_json({
                          {"ptr", reinterpret_cast<uint64_t>(ptr)},
                          {"size", num_bytes},
                          {"alignment", alignment},
                          {"allocator", Name()},
                      });
    }
}

void ForwardingAllocator::preDeallocation(void *ptr)
{
    LogAlloc() << "event: dealloc "
               << as_json({
                                     {"ptr", reinterpret_cast<uint64_t>(ptr)},
                                     {"size", RequestedSize(ptr)},
                                     {"allocator", Name()},
                                 });
}

void ForwardingAllocator::DeallocateRaw(void *ptr)
{
    preDeallocation(ptr);

    std::unordered_map<void *, size_t>::node_type nh;
    {
        auto g = sstl::with_guard(m_mu);

        // NOTE: This must be called under the lock, otherwise there is a data race on m_allocated
        // Consider the following order:
        // A: m_base->DeallocateRaw(12345)
        // B: ptr = m_base->AllocateRaw() = 12345
        // B: m_allocated[12345] = size         // Check failed!
        // A: m_allocated.extract(12345)
        m_base->DeallocateRaw(ptr);

        nh = m_allocated.extract(ptr);
    }
    postDeallocation(ptr);

    if (nh) {
        Unref();
    }
}

void ForwardingAllocator::postDeallocation(void *ptr)
{
    UNUSED(ptr);
}

bool ForwardingAllocator::TracksAllocationSizes()
{
    return true;
}

bool ForwardingAllocator::ShouldAllocateEmptyTensors()
{
    return m_base->ShouldAllocateEmptyTensors();
}

size_t ForwardingAllocator::RequestedSize(void *ptr)
{
    auto g = sstl::with_guard(m_mu);
    auto it = m_allocated.find(ptr);
    if (it == m_allocated.end()) {
        return 0;
    }
    return it->second;
}

size_t ForwardingAllocator::AllocatedSize(void *ptr)
{
    return m_base->AllocatedSize(ptr);
}

tf::int64 ForwardingAllocator::AllocationId(void *ptr)
{
    return m_base->AllocationId(ptr);
}

size_t ForwardingAllocator::AllocatedSizeSlow(void *ptr)
{
    return m_base->AllocatedSizeSlow(ptr);
}

void ForwardingAllocator::GetStats(tf::AllocatorStats *stats)
{
    m_base->GetStats(stats);
}

void ForwardingAllocator::recordSize(void *ptr, size_t size)
{
    auto g = sstl::with_guard(m_mu);
    if (!ptr) {
        // No enough memory
        return;
    }
    // Reference self for later deallocation
    Ref();

    CHECK(m_allocated.emplace(ptr, size).second) << "address already taken: " << as_hex(ptr);
}

/*static*/ tf::DeviceAttributes ShadowDevice::NewNameBase(const std::string &new_base, sstl::not_null<tf::Device *> base)
{
    tf::DeviceNameUtils::ParsedName parsed_name;
    CHECK(tf::DeviceNameUtils::ParseFullName(new_base, &parsed_name));
    tf::DeviceNameUtils::ParsedName underlying_parsed_name = base->parsed_name();
    CHECK(underlying_parsed_name.has_type);
    CHECK(underlying_parsed_name.has_id);
    parsed_name.type = underlying_parsed_name.type;
    parsed_name.id = underlying_parsed_name.id;
    auto name = tf::DeviceNameUtils::FullName(parsed_name.job, parsed_name.replica, parsed_name.task,
                                              parsed_name.type, parsed_name.id);
    auto attributes = base->attributes();
    attributes.set_name(name);

    return attributes;
}

/*static*/ std::unique_ptr<ShadowDevice> ShadowDevice::NewShadowDevice(const std::string &new_base,
                                                                       sstl::not_null<tf::Device *> base,
                                                                       bool isolateSessionState,
                                                                       bool ownsBase,
                                                                       CreateWrapperAllocatorFn fn)
{
    return std::make_unique<ShadowDevice>(base, NewNameBase(new_base, base), isolateSessionState, ownsBase, std::move(fn));
}

ShadowDevice::ShadowDevice(sstl::not_null<tf::Device *> base, const tf::DeviceAttributes &attr,
                           bool isolateSessionState, bool ownsBase, CreateWrapperAllocatorFn fn)
    : Device(base->env(), attr)
    , m_base(base)
    , m_isolate(isolateSessionState)
    , m_ownsBase(ownsBase)
    , m_createWrapperAllocator(std::move(fn))
{
    CHECK(m_createWrapperAllocator != nullptr);
    initialize();
}

void ShadowDevice::initialize()
{
    set_tensorflow_cpu_worker_threads(
        const_cast<CpuWorkerThreads *>(m_base->tensorflow_cpu_worker_threads()));
    set_tensorflow_gpu_device_info(const_cast<GpuDeviceInfo *>(m_base->tensorflow_gpu_device_info()));
    set_eigen_cpu_device(const_cast<Eigen::ThreadPoolDevice *>(m_base->eigen_cpu_device()));
#ifdef TENSORFLOW_USE_SYCL
    set_eigen_sycl_device(const_cast<Eigen::SyclDevice *>(m_base->eigen_sycl_device()));
#endif
}

ShadowDevice::~ShadowDevice()
{
    // First clear resource, which may still using allocators
    resource_manager()->Clear();
    // Then clear allocator wrappers
    clearWrapperCache();

    if (m_ownsBase) {
        delete m_base.get();
    }
}

tf::Allocator *ShadowDevice::GetAllocator(tf::AllocatorAttributes attr)
{
    auto a = m_base->GetAllocator(attr);
    return wrapAllocator(a, attr);
}

tf::Allocator *ShadowDevice::GetStepAllocator(tf::AllocatorAttributes attr,
                                              tf::ResourceMgr *step_resource_manager)
{
    auto a = m_base->GetStepAllocator(attr, step_resource_manager);
    return wrapAllocator(a, attr);
}

tf::Allocator *ShadowDevice::wrapAllocator(tf::Allocator *alloc, const tf::AllocatorAttributes &attr)
{
    AA key{alloc, attr};

    auto g = sstl::with_guard(m_mu);
    auto it = m_wrappedAllocators.find(key);
    if (it != m_wrappedAllocators.end()) {
        return it->second.get();
    }

    auto a = m_createWrapperAllocator(alloc, attr);
    auto pa = a.get();
    m_wrappedAllocators.emplace(key, std::move(a));
    return pa;
}

void ShadowDevice::clearWrapperCache()
{
    auto g = sstl::with_guard(m_mu);
    m_wrappedAllocators.clear();
}

bool ShadowDevice::RequiresRecordingAccessedTensors() const
{
    return m_base->RequiresRecordingAccessedTensors();
}

tf::PerOpGpuDevice *ShadowDevice::MakeGpuDevice()
{
    return m_base->MakeGpuDevice();
}

void ShadowDevice::ReinitializeGpuDevice(tf::OpKernelContext *context, tf::PerOpGpuDevice *device,
                                         tf::DeviceContext *dc, tf::Allocator *allocator)
{
    m_base->ReinitializeGpuDevice(context, device, dc, allocator);
}

tf::Status ShadowDevice::MakeTensorFromProto(const tf::TensorProto &tensor_proto,
                                             tf::AllocatorAttributes alloc_attrs, tf::Tensor *tensor)
{
    return m_base->MakeTensorFromProto(tensor_proto, alloc_attrs, tensor);
}

void ShadowDevice::Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context)
{
    m_base->Compute(op_kernel, context);
}

void ShadowDevice::ComputeAsync(tf::AsyncOpKernel *op_kernel, tf::OpKernelContext *context,
                                tf::AsyncOpKernel::DoneCallback done)
{
    m_base->ComputeAsync(op_kernel, context, std::move(done));
}

void ShadowDevice::ConsumeListOfAccessedTensors(tf::DeviceContext *context,
                                                const tf::TensorReferenceVector &tensors)
{
    m_base->ConsumeListOfAccessedTensors(context, tensors);
}

tf::Status ShadowDevice::Sync()
{
    return m_base->Sync();
}

tf::Status ShadowDevice::MaybeRewriteGraph(std::unique_ptr<tf::Graph> *graph)
{
    return m_base->MaybeRewriteGraph(graph);
}

tf::Status ShadowDevice::FillContextMap(const tf::Graph *graph, tf::DeviceContextMap *device_context_map)
{
    return m_base->FillContextMap(graph, device_context_map);
}

tf::ResourceMgr *ShadowDevice::resource_manager()
{
    if (m_isolate) {
        return Device::resource_manager();
    } else {
        return m_base->resource_manager();
    }
}

} // namespace salus::oplib::tensorflow
