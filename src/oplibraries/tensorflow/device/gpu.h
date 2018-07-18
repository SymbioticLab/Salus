//
// Created by peifeng on 3/22/18.
//

#ifndef SALUS_OPLIB_TENSORFLOW_DEVICE_GPU_H
#define SALUS_OPLIB_TENSORFLOW_DEVICE_GPU_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "utils/objectpool.h"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace salus::oplib::tensorflow {

using NodeStreamMap = std::unordered_map<int, int>;

class PerTaskGPUDevice;
class SalusGPUDevice : public ISalusDevice, public tf::BaseGPUDevice
{
public:
    SalusGPUDevice(const tf::SessionOptions &options, const std::string &name, tf::Bytes memory_limit,
                   const tf::DeviceLocality &locality, int gpu_id, const std::string &physical_device_desc,
                   tf::Allocator *gpu_allocator, tf::Allocator *cpu_allocator, int max_streams = 128);

    ~SalusGPUDevice() override = default;

    tf::Allocator *GetAllocator(tf::AllocatorAttributes attr) override;

    bool RequiresRecordingAccessedTensors() const override;

    void flushCacheFor(sstl::not_null<const tf::Graph *> graph) override;

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
    /**
     * @brief Try to allocate streams. May return less than requested
     *
     * @param num
     * @return
     */
    std::vector<int> allocateStreams(size_t num);

    /**
     * @brief Free streams
     * @param streams
     */
    void freeStreams(std::vector<int> &&streams);

    /**
     * @brief Get the device context correspond to stream `num'
     * @param num
     * @return
     */
    sstl::not_null<tf::DeviceContext *> deviceContext(int num) const
    {
        DCHECK_LT(num, static_cast<int>(device_contexts_.size()));
        return device_contexts_[num];
    }

    friend class PerTaskGPUDevice;
    std::shared_ptr<sstl::ObjectPool<PerTaskGPUDevice>> m_pool;

    std::mutex m_muStream;
    std::vector<bool> m_streamUsed;
};

class SalusGPUDeviceFactory : public tf::BaseGPUDeviceFactory
{
private:
    tf::BaseGPUDevice *CreateGPUDevice(const tf::SessionOptions &options, const std::string &name,
                                       tf::Bytes memory_limit, const tf::DeviceLocality &locality, int gpu_id,
                                       const std::string &physical_device_desc, tf::Allocator *gpu_allocator,
                                       tf::Allocator *cpu_allocator) override;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_DEVICE_GPU_H
