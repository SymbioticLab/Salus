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

#ifndef SALUS_OPLIB_TENSORFLOW_SALUSDEVICES
#define SALUS_OPLIB_TENSORFLOW_SALUSDEVICES

#include "execution/resources.h"
#include "oplibraries/tensorflow/v2/tfallocator.h"

#include <functional>
#include <memory>
#include <mutex>

namespace tensorflow {
class Device;
} // namespace tensorflow

class ResourceContext;

namespace salus::oplib::tensorflow {

class PerTaskDevice;
/**
 * @brief We use an extension to tensorflow devices.
 *
 * All TF devices should implement this interface.
 */
class ISalusDevice
{
public:
    virtual ~ISalusDevice() = default;

    virtual void flushCacheFor(sstl::not_null<const tf::Graph *> graph) = 0;

    virtual std::shared_ptr<PerTaskDevice> createPerTaskDevice(sstl::not_null<const tf::Graph *> g,
                                                               std::unique_ptr<ResourceContext> &&rctx) = 0;

    /**
     * @brief Get the tf::Device instance
     * @return
     */
    virtual tf::Device &as_tfdevice() = 0;
    virtual const tf::Device &as_tfdevice() const = 0;

    /**
     * @brief Safely cast a tf::Device to ISalusDeivce, w/o dynamic_cast and RTTI.
     *
     * It does this by first downcasting to concrete device type and then upcast,
     * based on the device name and type.
     *
     * @param device
     * @return
     */
    static ISalusDevice *safe_cast(tf::Device *device);
    static ISalusDevice &safe_cast(tf::Device &device);
};

/**
 * @brief Per task device knows the resource allocation for the particular task, it actually wrapps another
 * device
 *
 */
class PerTaskDevice : public tf::Device
{
public:
    explicit PerTaskDevice(sstl::not_null<tf::Device *> base, std::unique_ptr<ResourceContext> &&rctx);
    ~PerTaskDevice() override;

    virtual void reset(sstl::not_null<tf::Device *> base, std::unique_ptr<ResourceContext> &&rctx);

    ResourceContext &resourceContext() const
    {
        return *m_rctx;
    }

    template<typename T, typename = std::enable_if_t<std::is_convertible_v<T&, ISalusDevice&>>>
    T &underlayingDevice() const
    {
        return static_cast<T &>(ISalusDevice::safe_cast(underlayingDevice()));
    }

    tf::Device &underlayingDevice() const
    {
        return *m_base.get();
    }

    Resources failedResourceRequest() const;
    Resources peakResourceUsage() const;

    virtual tf::DeviceContext *deviceContextForNode(int id) const = 0;

    // Hook allocators
    tf::Allocator *GetAllocator(tf::AllocatorAttributes attr) override;
    tf::Allocator *GetStepAllocator(tf::AllocatorAttributes attr,
                                    tf::ResourceMgr *step_resource_manager) override;

    // Forwarding of DeviceBase methods
    bool RequiresRecordingAccessedTensors() const override;
    tf::PerOpGpuDevice *MakeGpuDevice() override;
    void ReinitializeGpuDevice(tf::OpKernelContext *context, tf::PerOpGpuDevice *device,
                               tf::DeviceContext *dc, tf::Allocator *allocator) override;
    tf::Status MakeTensorFromProto(const tf::TensorProto &tensor_proto,
                                   const tf::AllocatorAttributes alloc_attrs, tf::Tensor *tensor) override;

    // Forwarding of Device methods
    void Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context) override;
    void ComputeAsync(tf::AsyncOpKernel *op_kernel, tf::OpKernelContext *context,
                      tf::AsyncOpKernel::DoneCallback done) override;
    void ConsumeListOfAccessedTensors(tf::DeviceContext *context,
                                      const tf::TensorReferenceVector &tensors) override;
    tf::Status Sync() override;
    tf::Status MaybeRewriteGraph(std::unique_ptr<tf::Graph> *graph) override;
    tf::Status FillContextMap(const tf::Graph *graph, tf::DeviceContextMap *device_context_map) override;

private:
    void reinitialize();

    tf::Allocator *wrapAllocator(tf::Allocator *alloc, const tf::AllocatorAttributes &alloc_attrs);

    sstl::not_null<tf::Device *> m_base;
    std::shared_ptr<ResourceContext> m_rctx;

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
    std::unordered_map<AA, sstl::ScopedUnref<PerOpAllocator>, AAHasher> m_wrappedAllocators;
};

void maybeRegisterSalusDeviceFactories();

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_SALUSDEVICES
