//
// Created by peifeng on 3/22/18.
//

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/device/gpu.h"
#include "execution/executionengine.h"
#include "utils/threadutils.h"

#include <utility>

namespace salus::oplib::tensorflow {

class PerTaskGPUDevice : public PerTaskDevice
{
public:
    explicit PerTaskGPUDevice(SalusGPUDevice *base, std::unique_ptr<ResourceContext> &&rctx);

    void Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context) override;

    void ComputeAsync(tf::AsyncOpKernel *op_kernel, tf::OpKernelContext *context,
                      tf::AsyncOpKernel::DoneCallback done) override;

    tf::DeviceContext *deviceContextForNode(int id) const override;

    ~PerTaskGPUDevice() override;

private:
    sstl::ScopeGuards streamReleaser();

    std::vector<int> m_streams;
};

SalusGPUDevice::SalusGPUDevice(const tf::SessionOptions &options, const std::string &name,
                               tf::Bytes memory_limit, const tf::DeviceLocality &locality, int gpu_id,
                               const std::string &physical_device_desc, tf::Allocator *gpu_allocator,
                               tf::Allocator *cpu_allocator, int max_streams)
    : BaseGPUDevice(options, name, memory_limit, locality, gpu_id, physical_device_desc, gpu_allocator,
                    cpu_allocator, false /* sync every op */, max_streams)
    , m_streamUsed(static_cast<size_t>(max_streams), false)
    , m_streamAssignCache()
{
}

tf::Allocator *SalusGPUDevice::GetAllocator(tf::AllocatorAttributes attr)
{
    if (attr.on_host()) {
        if (attr.gpu_compatible()) {
            return tf::ProcessState::singleton()->GetCUDAHostAllocator(0);
        } else {
            return cpu_allocator_;
        }
    }
    return gpu_allocator_;
}

Status SalusGPUDevice::FillContextMap(const tf::Graph *graph,
                                      std::vector<tf::DeviceContext *> *device_context_map)
{
    UNUSED(device_context_map);

    VLOG(2) << "SalusGPUDevice::FillContextMap on " << name() << " for " << as_hex(graph);

    const auto num_streams = device_contexts_.size();

    NodeStreamMap *node_to_stream_id;
    {
        sstl::Guard g(m_muCache);
        if (m_streamAssignCache.count(graph) > 0) {
            LOG(WARNING) << "Detected graph address reuse: " << as_hex(graph);
        }
        node_to_stream_id = &m_streamAssignCache[graph];
    }

    // Special case for single stream.
    if (num_streams == 1) {
        return Status::OK();
    }

    tf::gpu_stream_util::AssignStreamsOpts opts;
    opts.max_streams = static_cast<int>(num_streams);
    TF_RETURN_IF_ERROR(tf::gpu_stream_util::AssignStreams(graph, opts, node_to_stream_id));

    VLOG(2) << "SalusGPUDevice::FillContextMap done";

    return Status::OK();
}

void SalusGPUDevice::flushCacheFor(const tf::Graph *graph)
{
    VLOG(2) << "SalusGPUDevice::flushCacheFor(" << as_hex(graph) << ") on " << name();
    sstl::Guard g(m_muCache);
    m_streamAssignCache.erase(graph);
}

std::unique_ptr<PerTaskDevice> SalusGPUDevice::createPerTaskDevice(const tf::Graph *graph,
                                                                   std::unique_ptr<ResourceContext> &&rctx)
{
    sstl::Guard g(m_muCache);
    VLOG(2) << "SalusGPUDevice::createPerTaskDevice on " << name() << " for " << as_hex(graph);
    return std::make_unique<PerTaskGPUDevice>(this, std::move(rctx));
}

std::vector<int> SalusGPUDevice::allocateStreams(size_t num)
{
    if (num == 0) {
        return {};
    }

    sstl::Guard g(m_muStream);
    std::vector<int> res;
    for (int i = 0; i != max_streams_; ++i) {
        if (!m_streamUsed[i]) {
            res.emplace_back(i);
            m_streamUsed[i] = true;
        }

        if (res.size() == num) {
            break;
        }
    }
    return res;
}

void SalusGPUDevice::freeStreams(std::vector<int> &&streams)
{
    if (streams.empty()) {
        return;
    }

    sstl::Guard g(m_muStream);
    for (auto i : streams) {
        m_streamUsed[i] = false;
    }
    streams.clear();
}

PerTaskGPUDevice::PerTaskGPUDevice(SalusGPUDevice *base, std::unique_ptr<ResourceContext> &&rctx)
    : PerTaskDevice(base, std::move(rctx))
    , m_streams()
{
    const auto numStreams = 1;
    auto &sdev = underlayingDevice<SalusGPUDevice>();

    if (auto scope = resourceContext().alloc(ResourceType::GPU_STREAM, numStreams)) {
        m_streams = sdev.allocateStreams(numStreams);
        if (m_streams.size() != numStreams) {
            LOG(ERROR) << "Can't get enough GPU streams, requested: " << numStreams << " got: " << m_streams.size();
            sdev.freeStreams(std::move(m_streams));
            scope.rollback();
        }
    }
}

tf::DeviceContext *PerTaskGPUDevice::deviceContextForNode(int id) const
{
    UNUSED(id);
    if (m_streams.empty()) {
        LOG(ERROR) << "No GPU streams available for device context";
        return nullptr;
    }

    auto stream = m_streams[0];
    auto &sdev = underlayingDevice<SalusGPUDevice>();

    DCHECK_LT(stream, static_cast<int>(sdev.device_contexts_.size()));
    return sdev.device_contexts_[stream];
}

sstl::ScopeGuards PerTaskGPUDevice::streamReleaser()
{
    // release stream when finish
    return sstl::ScopeGuards([this](){
        if (auto num = m_streams.size()) {
            underlayingDevice<SalusGPUDevice>().freeStreams(std::move(m_streams));
            resourceContext().dealloc(ResourceType::GPU_STREAM, num);
        }
    });
}

void PerTaskGPUDevice::Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context)
{
    auto sr = streamReleaser();
    PerTaskDevice::Compute(op_kernel, context);
}

void PerTaskGPUDevice::ComputeAsync(tf::AsyncOpKernel *op_kernel, tf::OpKernelContext *context,
                                    tf::AsyncOpKernel::DoneCallback done)
{
    auto sr = streamReleaser();
    PerTaskDevice::ComputeAsync(op_kernel, context, done);
}

PerTaskGPUDevice::~PerTaskGPUDevice()
{
    auto sr = streamReleaser();
}


tf::BaseGPUDevice *SalusGPUDeviceFactory::CreateGPUDevice(const tf::SessionOptions &options,
                                                          const std::string &name, tf::Bytes memory_limit,
                                                          const tf::DeviceLocality &locality, int gpu_id,
                                                          const std::string &physical_device_desc,
                                                          tf::Allocator *gpu_allocator,
                                                          tf::Allocator *cpu_allocator)
{
    // TODO: detect maximum streams in GPU
    auto max_streams = 128;

    auto dev = new SalusGPUDevice(options, name, memory_limit, locality, gpu_id, physical_device_desc,
                                  gpu_allocator, cpu_allocator, max_streams);
    VLOG(1) << "Creating SalusGPUDevice " << as_hex(dev) << " which is a tf::Device "
            << as_hex(static_cast<tf::Device *>(dev)) << " and also a ISalusDevice "
            << as_hex(static_cast<ISalusDevice *>(dev));
    return dev;
}

} // namespace salus::oplib::tensorflow
