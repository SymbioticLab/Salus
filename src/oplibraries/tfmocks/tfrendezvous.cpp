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

#include "tfrendezvous.h"
#include "tfsession.h"

#include "platform/logging.h"
#include "utils/macros.h"

#include <tensorflow/core/common_runtime/copy_tensor.h>
#include <tensorflow/core/common_runtime/device.h>
#include <tensorflow/core/util/device_name_utils.h>

using tensorflow::Status;
using tensorflow::Tensor;

namespace {
bool isSameDevice(const tensorflow::DeviceNameUtils::ParsedName &a,
                  const tensorflow::DeviceNameUtils::ParsedName &b)
{
    return tensorflow::DeviceNameUtils::IsSameAddressSpace(a, b)
           && (a.has_type && b.has_type && (a.type == b.type)) && (a.has_id && b.has_id && (a.id == b.id));
}

class WrapperDeviceContext : public tensorflow::DeviceContext
{
    tensorflow::Device *m_device;
    tensorflow::DeviceContext *m_actualCtx;

public:
    // Takes one ref on `actual`
    WrapperDeviceContext(tensorflow::Device *dev, tensorflow::DeviceContext *actual)
        : m_device(dev)
        , m_actualCtx(actual)
    {
        assert(m_actualCtx);
    }

    ~WrapperDeviceContext() override
    {
        m_actualCtx->Unref();
    }

    perftools::gputools::Stream *stream() const override
    {
        return m_actualCtx->stream();
    }

    void MaintainLifetimeOnStream(const tensorflow::Tensor *t,
                                  perftools::gputools::Stream *stream) const override
    {
        return m_actualCtx->MaintainLifetimeOnStream(t, stream);
    }

    void CopyCPUTensorToDevice(const tensorflow::Tensor *cpu_tensor, tensorflow::Device *device,
                               tensorflow::Tensor *device_tensor,
                               tensorflow::StatusCallback done) const override
    {
        return m_actualCtx->CopyCPUTensorToDevice(cpu_tensor, device, device_tensor, done);
    }

    void CopyDeviceTensorToCPU(const tensorflow::Tensor *device_tensor, tensorflow::StringPiece tensor_name,
                               tensorflow::Device *device, tensorflow::Tensor *cpu_tensor,
                               tensorflow::StatusCallback done) override
    {
        return m_actualCtx->CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor, done);
    }

    tensorflow::Device *device() const
    {
        return m_device;
    }

    tensorflow::DeviceContext *wrapped() const
    {
        return m_actualCtx;
    }
};
} // namespace

TFRendezvous::SendItem::SendItem()
    : SendItem(Args(), false, tensorflow::Tensor())
{
}

TFRendezvous::SendItem::SendItem(const Args &a, bool d, tensorflow::Tensor &&v)
    : args(a)
    , isDead(d)
    , val(std::move(v))
{
}

TFRendezvous::RecvItem::RecvItem()
    : RecvItem(Args())
{
}

TFRendezvous::RecvItem::RecvItem(const Args &a)
    : args(a)
{
}

TFRendezvous::TFRendezvous(TFContext *tfctx)
    : m_tfctx(tfctx)
    , m_local(m_tfctx->execState->rendez())
{
    m_local->Ref();
}

TFRendezvous::~TFRendezvous()
{
    m_local->Unref();
}
tensorflow::Status TFRendezvous::triggerSend(const ParsedKey &parsed, const Args &send_args,
                                             const tensorflow::Tensor &val, const bool is_dead)
{
    return m_local->Send(parsed, send_args, val, is_dead);
}

tensorflow::Status TFRendezvous::Send(const ParsedKey &parsed, const Args &send_args,
                                      const tensorflow::Tensor &val, const bool is_dead)
{
    INFO("TFRendezvous::Send {}", parsed.FullKey().ToString());

    if (send_args.device_context) {
        send_args.device_context->Ref();
    }

    if (!isSameDevice(parsed.src, parsed.dst)) {
        auto key = parsed.FullKey().ToString();
        auto it = m_tensors.end();
        {
            tensorflow::mutex_lock locker(m_mu);
            it = m_tensors.find(key);
            if (it != m_tensors.end()) {
                ERR("Duplicated send: {}", key);
                return tensorflow::errors::Aborted("Duplicated send: ", parsed.FullKey());
            }
            tensorflow::Tensor copy(val);
            m_tensors.emplace(
                std::make_pair(parsed.FullKey().ToString(), SendItem{send_args, is_dead, std::move(copy)}));
        }
    }

    auto args = send_args;
    args.device_context =
        new WrapperDeviceContext(static_cast<tensorflow::Device *>(m_tfctx->ctx()->device()),
                                 send_args.device_context);

    return m_local->Send(parsed, args, val, is_dead);
}

void TFRendezvous::RecvAsync(const ParsedKey &parsed, const Args &recv_args, DoneCallback done)
{
    INFO("TFRendezvous::RecvAsync {}", parsed.FullKey().ToString());

    if (recv_args.device_context) {
        recv_args.device_context->Ref();
    }

    if (!isSameDevice(parsed.src, parsed.dst)) {
        // Pending recv must be fullfilled later by RPC
        INFO("Saving recv request for later RPC");
        auto key = parsed.FullKey().ToString();
        DEBUG("Pending recv request saved: {}", key);
        {
            tensorflow::mutex_lock locker(m_recvmu);
            auto it = m_recv.find(key);
            if (it != m_recv.end()) {
                ERR("Duplicated recv: {}", key);
                done(tensorflow::errors::Internal("Duplicated recv"), Args(), recv_args, tensorflow::Tensor(),
                     false);
                return;
            }
            m_recv.emplace(std::make_pair(parsed.FullKey().ToString(), RecvItem{recv_args}));
        }
    }

    auto args = recv_args;
    args.device_context =
        new WrapperDeviceContext(static_cast<tensorflow::Device *>(m_tfctx->ctx()->device()),
                                 recv_args.device_context);

    m_local->RecvAsync(parsed, args, [this, parsed, done](auto status, auto send_args, auto recv_args,
                                                          auto in, auto is_dead) {
        if (!isSameDevice(parsed.src, parsed.dst)) {
            // this is triggered by triggerSend, meaning this is a rendez between outside and rpc.
            // simply return.
            done(status, send_args, recv_args, in, is_dead);
            return;
        }

        auto send_wrapper = static_cast<WrapperDeviceContext *>(send_args.device_context);
        auto recv_wrapper = static_cast<WrapperDeviceContext *>(recv_args.device_context);
        assert(send_wrapper);
        assert(recv_wrapper);

        // If "in" is an uninitialized tensor, do copy-construction to preserve
        // the uninitialized state, along with data type and shape info, which
        // is useful for debugger purposes.
        auto out = in.IsInitialized() ? new tensorflow::Tensor : new tensorflow::Tensor(in);

        auto actual_send_args = send_args;
        auto actual_recv_args = recv_args;
        actual_send_args.device_context = send_wrapper->wrapped();
        actual_recv_args.device_context = recv_wrapper->wrapped();

        auto final_callback = [done, actual_send_args, actual_recv_args, out, is_dead](auto s) {
            done(s, actual_send_args, actual_recv_args, *out, is_dead);
            delete out;
        };

        if (status.ok() && in.IsInitialized()) {
            SameWorkerRecvDone(parsed, send_wrapper->device(), recv_wrapper->device(), actual_send_args,
                               actual_recv_args, in, out, final_callback);
        } else {
            final_callback(status);
        }
    });
}

void TFRendezvous::SameWorkerRecvDone(const ParsedKey &parsed, tensorflow::Device *send_dev,
                                      tensorflow::Device *recv_dev, const Args &send_args,
                                      const Args &recv_args, const tensorflow::Tensor &in,
                                      tensorflow::Tensor *out, StatusCallback done)
{
    INFO("TFRendezvous: copying tensor for rendezvous key {}, actual send device {}, actual recv device {}",
         parsed.FullKey(), send_dev->name(), recv_dev->name());

    // Do a quick copy (sharing the underlying buffer) if both tensors
    // are on host memory.
    const bool src_host =
        (send_args.alloc_attrs.on_host() || send_dev->attributes().device_type() == tensorflow::DEVICE_CPU);
    const bool dst_host =
        (recv_args.alloc_attrs.on_host() || recv_dev->attributes().device_type() == tensorflow::DEVICE_CPU);
    if (src_host && dst_host) {
        *out = in;
        done(Status::OK());
        return;
    }

    // This copy must involve a non-CPU device. Hence, "in" must support DMA
    // (e.g., string tensors do not work on GPU).
    if (!tensorflow::DataTypeCanUseMemcpy(in.dtype())) {
        done(tensorflow::errors::InvalidArgument("Non-DMA-safe ", tensorflow::DataTypeString(in.dtype()),
                                                 " tensor may not be copied from/to a GPU."));
        return;
    }

    auto attr = recv_args.alloc_attrs;
    attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() || recv_args.alloc_attrs.gpu_compatible());
    auto out_allocator = recv_dev->GetAllocator(attr);
    tensorflow::Tensor copy(out_allocator, in.dtype(), in.shape());
    *out = copy;

    tensorflow::CopyTensor::ViaDMA(parsed.edge_name, send_args.device_context, recv_args.device_context,
                                   send_dev, recv_dev, send_args.alloc_attrs, recv_args.alloc_attrs, &in, out,
                                   done);
}

void TFRendezvous::StartAbort(const tensorflow::Status &status)
{
    return m_local->StartAbort(status);
}

TFRendezvous::SentTensorTable TFRendezvous::releasePendingSentTensors()
{
    SentTensorTable table;
    tensorflow::mutex_lock locker(m_mu);
    std::swap(table, m_tensors);
    return table;
}

TFRendezvous::RecvTable TFRendezvous::releasePendingRecv()
{
    RecvTable table;
    tensorflow::mutex_lock locker(m_recvmu);
    std::swap(table, m_recv);
    return table;
}
