//
// Created by peifeng on 3/22/18.
//

#ifndef SALUS_OPLIB_TENSORFLOW_DEVICE_CPU_H
#define SALUS_OPLIB_TENSORFLOW_DEVICE_CPU_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "utils/pointerutils.h"
#include "utils/objectpool.h"

namespace salus::oplib::tensorflow {

class PerTaskCPUDevice;
class SalusCPUDevice : public ISalusDevice, public tf::LocalDevice
{
public:
    SalusCPUDevice(const tf::SessionOptions &options, const std::string &name, tf::Bytes memory_limit,
                   const tf::DeviceLocality &locality, tf::Allocator *allocator);

    ~SalusCPUDevice() override = default;

    tf::Allocator *GetAllocator(tf::AllocatorAttributes attr) override;

    Status Sync() override
    {
        return Status::OK();
    }

    void Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context) override
    {
        op_kernel->Compute(context);
    }

    Status MakeTensorFromProto(const tf::TensorProto &tensor_proto, const tf::AllocatorAttributes alloc_attrs,
                               tf::Tensor *tensor) override;

    void flushCacheFor(sstl::not_null<const tf::Graph *>) override
    {
    }

    std::shared_ptr<PerTaskDevice> createPerTaskDevice(sstl::not_null<const tf::Graph *> graph,
                                                       std::unique_ptr<ResourceContext> &&rctx) override;

    tf::Device &as_tfdevice() override
    {
        return *this;
    }

    const tf::Device &as_tfdevice() const override
    {
        return *this;
    }

private:
    sstl::not_null<tf::Allocator *> m_allocator; // not owned

    std::shared_ptr<sstl::ObjectPool<PerTaskCPUDevice>> m_pool;
};

class SalusCPUDeviceFactory : public tf::DeviceFactory
{
public:
    Status CreateDevices(const tf::SessionOptions &options, const std::string &name_prefix,
                         std::vector<tf::Device *> *devices) override;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_DEVICE_CPU_H
