//
// Created by peifeng on 3/22/18.
//

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/device/cpu.h"
#include "execution/executionengine.h"
#include "utils/threadutils.h"
#include "utils/objectpool.h"

namespace salus::oplib::tensorflow {

class PerTaskCPUDevice : public PerTaskDevice
{
public:
    explicit PerTaskCPUDevice(tf::Device *base, std::unique_ptr<ResourceContext> &&rctx)
        : PerTaskDevice(base, std::move(rctx))
    {
    }

    tf::DeviceContext *deviceContextForNode(int) const override
    {
        return nullptr;
    }
};

SalusCPUDevice::SalusCPUDevice(const tf::SessionOptions &options, const std::string &name,
                               tf::Bytes memory_limit, const tf::DeviceLocality &locality,
                               tf::Allocator *allocator)
    : LocalDevice(options, tf::Device::BuildDeviceAttributes(name, tf::DEVICE_CPU, memory_limit, locality))
    , m_allocator(allocator)
{
}

tf::Allocator *SalusCPUDevice::GetAllocator(tf::AllocatorAttributes attr)
{
    if (attr.gpu_compatible()) {
        return tf::ProcessState::singleton()->GetCUDAHostAllocator(0);
    }
    return m_allocator;
}

Status SalusCPUDevice::MakeTensorFromProto(const tf::TensorProto &tensor_proto,
                                           const tf::AllocatorAttributes alloc_attrs, tf::Tensor *tensor)
{
    UNUSED(alloc_attrs);
    if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= tf::DataType_MAX) {
        tf::Tensor parsed(tensor_proto.dtype());
        if (parsed.FromProto(m_allocator, tensor_proto)) {
            *tensor = std::move(parsed);
            return Status::OK();
        }
    }
    return tf::errors::InvalidArgument("Cannot parse tensor from proto: ",
                                       tf::ProtoDebugString(tensor_proto));
}

std::shared_ptr<PerTaskDevice> SalusCPUDevice::createPerTaskDevice(const tf::Graph *graph,
                                                                   std::unique_ptr<ResourceContext> &&rctx)
{
    UNUSED(graph);
    thread_local sstl::ObjectPool<PerTaskCPUDevice> pool;
    return pool.acquire(this, std::move(rctx));
}

Status SalusCPUDeviceFactory::CreateDevices(const tf::SessionOptions &options, const std::string &name_prefix,
                                            std::vector<tf::Device *> *devices)
{
    // TODO(zhifengc/tucker): Figure out the number of available CPUs and/or NUMA configuration.
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
        n = iter->second;
    }
    for (int i = 0; i < n; i++) {
        auto name = tf::strings::StrCat(name_prefix, "/cpu:", i);
        // use tf::cpu_allocator to select from cpu allocatory registary
        auto dev = new SalusCPUDevice(options, name, tf::Bytes(256 << 20), {}, tf::cpu_allocator());
        VLOG(1) << "Creating SalusCPUDevice " << as_hex(dev) << " which is a tf::Device "
                << as_hex(static_cast<tf::Device *>(dev)) << " and also a ISalusDevice "
                << as_hex(static_cast<ISalusDevice *>(dev));
        devices->push_back(dev);
    }

    return Status::OK();
}

} // namespace salus::oplib::tensorflow
