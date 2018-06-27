//
// Created by peifeng on 3/22/18.
//

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/device/gpu.h"
#include "execution/engine/resourcecontext.h"
#include "utils/threadutils.h"

#include <utility>

namespace salus::oplib::tensorflow {

class PerTaskGPUDevice : public PerTaskDevice
{
public:
    explicit PerTaskGPUDevice(sstl::not_null<tf::Device *> base, std::unique_ptr<ResourceContext> &&rctx);

    void reset(sstl::not_null<tf::Device *> base, std::unique_ptr<ResourceContext> &&rctx) override;

    void Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context) override;

    void ComputeAsync(tf::AsyncOpKernel *op_kernel, tf::OpKernelContext *context,
                      tf::AsyncOpKernel::DoneCallback done) override;

    tf::DeviceContext *deviceContextForNode(int id, bool isAsync) override;

    ~PerTaskGPUDevice() override;

private:
    void requestStreams();

private:
    sstl::ScopeGuards useStreams();
    void releaseStreams();

    std::vector<int> m_streams;
};

SalusGPUDevice::SalusGPUDevice(const tf::SessionOptions &options, const std::string &name,
                               tf::Bytes memory_limit, const tf::DeviceLocality &locality, int gpu_id,
                               const std::string &physical_device_desc, tf::Allocator *gpu_allocator,
                               tf::Allocator *cpu_allocator, int max_streams)
    : BaseGPUDevice(options, name, memory_limit, locality, gpu_id, physical_device_desc, gpu_allocator,
                    cpu_allocator, false /* sync every op */, max_streams)
    , m_pool(std::make_shared<sstl::ObjectPool<PerTaskGPUDevice>>())
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

bool SalusGPUDevice::RequiresRecordingAccessedTensors() const
{
    return BaseGPUDevice::RequiresRecordingAccessedTensors();
}

Status SalusGPUDevice::FillContextMap(const tf::Graph *, std::vector<tf::DeviceContext *> *)
{
    /*
    VLOG(3) << "SalusGPUDevice::FillContextMap on " << name() << " for " << as_hex(graph);
    const auto num_streams = device_contexts_.size();

    NodeStreamMap *node_to_stream_id;
    {
        auto g = sstl::with_guard(m_muCache);
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

    VLOG(3) << "SalusGPUDevice::FillContextMap done";
     */
    return Status::OK();
}

void SalusGPUDevice::flushCacheFor(sstl::not_null<const tf::Graph *>)
{
//    VLOG(3) << "SalusGPUDevice::flushCacheFor(" << as_hex(graph) << ") on " << name();
//    auto g = sstl::with_guard(m_muCache);
//    m_streamAssignCache.erase(graph);
}

std::shared_ptr<PerTaskDevice> SalusGPUDevice::createPerTaskDevice(sstl::not_null<const tf::Graph *> graph,
                                                                   std::unique_ptr<ResourceContext> &&rctx)
{
    VLOG(3) << "SalusGPUDevice::createPerTaskDevice on " << name() << " for " << as_hex(graph.get());
    return m_pool->acquire(this, std::move(rctx));
}

std::vector<int> SalusGPUDevice::allocateStreams(size_t num)
{
    if (num == 0) {
        return {};
    }

    auto g = sstl::with_guard(m_muStream);
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

    auto g = sstl::with_guard(m_muStream);
    for (auto i : streams) {
        m_streamUsed[i] = false;
    }
    streams.clear();
}

PerTaskGPUDevice::PerTaskGPUDevice(sstl::not_null<tf::Device *> base, std::unique_ptr<ResourceContext> &&rctx)
    : PerTaskDevice(base, std::move(rctx))
{
}

void PerTaskGPUDevice::reset(sstl::not_null<tf::Device *> base, std::unique_ptr<ResourceContext> &&rctx)
{
    releaseStreams();
    PerTaskDevice::reset(base, std::move(rctx));
}

void PerTaskGPUDevice::requestStreams()
{
    if (!m_streams.empty()) {
        LOG(WARNING) << "Duplicate call to PerTaskGPUDevice::requestStreams";
        return;
    }

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

tf::DeviceContext *PerTaskGPUDevice::deviceContextForNode(int id, bool isAsync)
{
    UNUSED(id);

#if defined(SALUS_ENABLE_STATIC_STREAM)
    UNUSED(isAsync);

    // use a round robin to assign streams to session
    static std::mutex mu;
    static std::unordered_map<std::string, int> seenSessions;
    static auto nextStream = 0;

    auto stream = 0;
    {
        auto g = sstl::with_guard(mu);

        auto [it, newSession] = seenSessions.try_emplace(resourceContext().sessHandle, nextStream);
        if (newSession) {
            nextStream = (nextStream + 1) % 128;
        }
        stream = it->second;
    }
#else
    if (!isAsync) {
        requestStreams();
    }

    // use default stream if we have none
    auto stream = m_streams.empty() ? 0 : m_streams[0];
#endif

    return underlayingDevice<SalusGPUDevice>().deviceContext(stream).get();
}

void PerTaskGPUDevice::releaseStreams()
{
    VLOG(3) << "Releasing GPU stream: " << m_streams;
    if (auto num = m_streams.size()) {
        underlayingDevice<SalusGPUDevice>().freeStreams(std::move(m_streams));
        resourceContext().dealloc(ResourceType::GPU_STREAM, num);
    }
}

sstl::ScopeGuards PerTaskGPUDevice::useStreams()
{
    // release stream when finish
    return sstl::ScopeGuards([this](){
        releaseStreams();
    });
}

void PerTaskGPUDevice::Compute(tf::OpKernel *op_kernel, tf::OpKernelContext *context)
{
#if !defined(SALUS_ENABLE_STATIC_STREAM)
    if (m_streams.empty()) {
        LOG(ERROR) << "No GPU streams available for " << op_kernel->name() << " using default one";
    }
    auto sr = useStreams();
#endif
    PerTaskDevice::Compute(op_kernel, context);
}

void PerTaskGPUDevice::ComputeAsync(tf::AsyncOpKernel *op_kernel, tf::OpKernelContext *context,
                                    tf::AsyncOpKernel::DoneCallback done)
{
    CHECK(m_streams.empty());
    // Async compute don't need stream
    PerTaskDevice::ComputeAsync(op_kernel, context, done);
}

PerTaskGPUDevice::~PerTaskGPUDevice()
{
#if !defined(SALUS_ENABLE_STATIC_STREAM)
    releaseStreams();
#endif
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
    VLOG(3) << "Creating SalusGPUDevice " << as_hex(dev) << " which is a tf::Device "
            << as_hex(static_cast<tf::Device *>(dev)) << " and also a ISalusDevice "
            << as_hex(static_cast<ISalusDevice *>(dev));
    return dev;
}

} // namespace salus::oplib::tensorflow
