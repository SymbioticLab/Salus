/*
 * Copyright (c) 2018, peifeng <email>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "oplibraries/tensorflow/device/gpu/sessiondevice.h"

#include "oplibraries/tensorflow/device/sessionallocator.h"
#include "utils/pointerutils.h"

#include <string>

namespace salus::oplib::tensorflow {

SessionDevice::SessionDevice(sstl::not_null<tf::Device *> base, const std::string &newBaseName, std::string sessHandle,
                             GpuDeviceInfo newInfo, std::vector<StreamAndContext> streams)
    : ShadowDevice(base, NewNameBase(newBaseName, base),
                   /*isolateSessionState = */ true, /*ownsBase = */ false,
                   [this](auto alloc, auto &&attrs) {
                       return createWrappedAllocator(alloc, std::forward<decltype(attrs)>(attrs));
                   })
    , m_sessHandle(std::move(sessHandle))
    , m_gpuDeviceInfo(newInfo)
    , m_streams(std::move(streams))
{
    DCHECK(!m_streams.empty());

    set_tensorflow_gpu_device_info(&m_gpuDeviceInfo);
}

sstl::ScopedUnref<ForwardingAllocator> SessionDevice::createWrappedAllocator(tf::Allocator *alloc,
                                                                             const tf::AllocatorAttributes &)
{
    return sstl::make_scoped_unref<SessionAllocator>(m_sessHandle, sstl::not_null{alloc});
}

tf::Status SessionDevice::FillContextMap(const tf::Graph *, tf::DeviceContextMap *)
{
    // Force the use of default stream and device context set in tensorflow_gpu_device_info
    return Status::OK();
}

tf::Status SessionDevice::Sync()
{
    // Only sync the main default stream
    tf::Notification n[4];
    tensorflow_gpu_device_info()->event_mgr->ThenExecute(m_streams[0].first->compute, [&n](){
        n[0].Notify();
    });
    tensorflow_gpu_device_info()->event_mgr->ThenExecute(m_streams[0].first->host_to_device, [&n](){
        n[1].Notify();
    });
    tensorflow_gpu_device_info()->event_mgr->ThenExecute(m_streams[0].first->device_to_host, [&n](){
        n[2].Notify();
    });
    tensorflow_gpu_device_info()->event_mgr->ThenExecute(m_streams[0].first->device_to_device, [&n](){
        n[3].Notify();
    });

    n[0].WaitForNotification();
    n[1].WaitForNotification();
    n[2].WaitForNotification();
    n[3].WaitForNotification();
    return Status::OK();

    /*
    sstl::semaphore sa;
    for (auto &[sg, ctx] : m_streams) {
        UNUSED(ctx);
        tensorflow_gpu_device_info()->event_mgr->ThenExecute(sg->compute, [&sa](){
            sa.notify();
        });
        tensorflow_gpu_device_info()->event_mgr->ThenExecute(sg->host_to_device, [&sa](){
            sa.notify();
        });
        tensorflow_gpu_device_info()->event_mgr->ThenExecute(sg->device_to_host, [&sa](){
            sa.notify();
        });
        tensorflow_gpu_device_info()->event_mgr->ThenExecute(sg->device_to_device, [&sa](){
            sa.notify();
        });
    }

    const constexpr int PerGroupStreams = 4;
    sa.wait(static_cast<uint32_t>(m_streams.size() * PerGroupStreams));
    return Status::OK();
     */
}

} // namespace salus::oplib::tensorflow
