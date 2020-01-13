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

#ifndef SALUS_OPLIB_TENSORFLOW_DEVICEWRAPPERS_H
#define SALUS_OPLIB_TENSORFLOW_DEVICEWRAPPERS_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "utils/pointerutils.h"
#include "utils/macros.h"

#include <functional>

namespace salus::oplib::tensorflow {

/**
 * @brief An allocator that does nothing but forwards the call to underlaying allocator
 */
class ForwardingAllocator : public tf::Allocator, public tf::core::RefCounted
{
    const sstl::not_null<tf::Allocator *> m_base;

    const std::string m_prefix;

    mutable std::mutex m_mu;
    std::unordered_map<void *, size_t> m_allocated GUARDED_BY(m_mu);

public:
    explicit ForwardingAllocator(sstl::not_null<tf::Allocator *> actual, const std::string &namePrefix = "");

    ~ForwardingAllocator() override;

    sstl::not_null<tf::Allocator *> base() const
    {
        return m_base;
    }

    std::string Name() override;

    void *AllocateRaw(size_t alignment, size_t num_bytes) override;

    void *AllocateRaw(size_t alignment, size_t num_bytes,
                      const tf::AllocationAttributes &allocation_attr) override;

    void DeallocateRaw(void *ptr) override;

    bool TracksAllocationSizes() override;

    bool ShouldAllocateEmptyTensors() override;

    size_t RequestedSize(void *ptr) override;

    size_t AllocatedSize(void *ptr) override;

    tf::int64 AllocationId(void *ptr) override;

    size_t AllocatedSizeSlow(void *ptr) override;

    void GetStats(tf::AllocatorStats *stats) override;

protected:
    virtual bool preAllocation(size_t alignment, size_t num_bytes,
                               const tf::AllocationAttributes &allocation_attr);
    virtual void postAllocation(void *ptr, size_t alignment, size_t num_bytes,
                                const tf::AllocationAttributes &allocation_attr);
    virtual void preDeallocation(void *ptr);
    virtual void postDeallocation(void *ptr);

    void recordSize(void *ptr, size_t size);
};

inline sstl::ScopedUnref<ForwardingAllocator> NewForwardingAllocator(tf::Allocator *alloc, const tf::AllocatorAttributes &)
{
    return sstl::make_scoped_unref<ForwardingAllocator>(alloc);
}

/**
 * @brief Provides a hook point to insert custom wrapped allocators.
 */
class ShadowDevice : public tf::Device
{
public:
    using CreateWrapperAllocatorFn =
        std::function<sstl::ScopedUnref<ForwardingAllocator>(tf::Allocator *,
                                                             const tf::AllocatorAttributes &)>;

    static tf::DeviceAttributes NewNameBase(const std::string &new_base, sstl::not_null<tf::Device *> base);

    static std::unique_ptr<ShadowDevice> NewShadowDevice(const std::string &new_base, sstl::not_null<tf::Device *> base,
                                                         bool isolateSessionState = true, bool ownsBase = false,
                                                         CreateWrapperAllocatorFn fn = NewForwardingAllocator);

    explicit ShadowDevice(sstl::not_null<tf::Device *> base, const tf::DeviceAttributes &attr,
                          bool isolateSessionState, bool ownsBase, CreateWrapperAllocatorFn fn);

    ~ShadowDevice() override;

    sstl::not_null<tf::Device *> base() const { return m_base; }

    // Hook allocators
    tf::Allocator *GetAllocator(tf::AllocatorAttributes attr) override;
    tf::Allocator *GetStepAllocator(tf::AllocatorAttributes attr,
                                    tf::ResourceMgr *step_resource_manager) override;

    // Forwarding of DeviceBase methods
    bool RequiresRecordingAccessedTensors() const override;
    tf::PerOpGpuDevice *MakeGpuDevice() override;
    void ReinitializeGpuDevice(tf::OpKernelContext *context, tf::PerOpGpuDevice *device,
                               tf::DeviceContext *dc, tf::Allocator *allocator) override;
    tf::Status MakeTensorFromProto(const tf::TensorProto &tensor_proto, tf::AllocatorAttributes alloc_attrs,
                                   tf::Tensor *tensor) override;

    // Forwarding of Device methods
    void Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context) override;
    void ComputeAsync(tf::AsyncOpKernel *op_kernel, tf::OpKernelContext *context,
                      tf::AsyncOpKernel::DoneCallback done) override;
    void ConsumeListOfAccessedTensors(tf::DeviceContext *context,
                                      const tf::TensorReferenceVector &tensors) override;
    tf::Status Sync() override;
    tf::Status MaybeRewriteGraph(std::unique_ptr<tf::Graph> *graph) override;
    tf::Status FillContextMap(const tf::Graph *graph, tf::DeviceContextMap *device_context_map) override;
    tf::ResourceMgr *resource_manager() override;

    SALUS_DISALLOW_COPY_AND_ASSIGN(ShadowDevice);
protected:
    /**
     * @brief Get a wrapped allocator, use a cached one if present
     * @param alloc
     * @param alloc_attrs
     * @return
     */
    tf::Allocator *wrapAllocator(tf::Allocator *alloc, const tf::AllocatorAttributes &alloc_attrs);

    /**
     * @brief Clear all cached wrapped allocators
     */
    void clearWrapperCache();

    /**
     * @brief Initialize with the base device
     */
    void initialize();

private:
    const sstl::not_null<tf::Device *> m_base;
    const bool m_isolate;
    const bool m_ownsBase;
    const CreateWrapperAllocatorFn m_createWrapperAllocator;

    mutable std::mutex m_mu;
    struct AA
    {
        tf::Allocator *alloc;
        tf::AllocatorAttributes attr;

        bool operator==(const AA &other) const
        {
            return alloc == other.alloc && attr.value == other.attr.value;
        }
    };
    struct AAHasher
    {
        size_t operator()(const AA &aa) const
        {
            using std::hash;
            size_t val = 0;
            sstl::hash_combine(val, aa.alloc);
            sstl::hash_combine(val, aa.attr.value);
            return val;
        }
    };
    std::unordered_map<AA, sstl::ScopedUnref<ForwardingAllocator>, AAHasher> m_wrappedAllocators
        GUARDED_BY(m_mu);
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_DEVICEWRAPPERS_H
