/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Peifeng Yu <peifeng@umich.edu>
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
 */

#include "rendezvousmgr.h"
#include "oplibraries/tensorflow/worker/devicecontextwithdevice.h"
#include "utils/threadutils.h"

namespace tf = ::tensorflow;

namespace salus::oplib::tensorflow {

tf::BaseRemoteRendezvous *SalusRendezvousMgr::Create(tf::int64 step_id, const tf::WorkerEnv *worker_env)
{
    return new WorkerRendezvous(worker_env, step_id);
}

bool WorkerRendezvous::FindTensor(const std::string &key, tf::Tensor &t)
{
    auto l = sstl::with_guard(m_mu);
    auto it = m_tensors.find(key);
    if (it != m_tensors.end()) {
        t = it->second;
        return true;
    }
    return false;
}

tf::Status WorkerRendezvous::Send(const ParsedKey &key, const Args &args, const tf::Tensor &val,
                                  const bool is_dead)
{
    // Must not hold lock when calling Send, which in turn may call cb, and requiring lock again
    {
        auto l = sstl::with_guard(m_mu);
        m_tensors.emplace(key.FullKey().ToString(), val);
    }
    return BaseRemoteRendezvous::Send(key, args, val, is_dead);
}

void WorkerRendezvous::RecvAsync(const ParsedKey &key, const Args &args, DoneCallback done)
{
    auto full_key = key.FullKey().ToString();
    auto final_done = [done = std::move(done), full_key, this](auto s, auto send_args, auto recv_args,
                                                               auto val, auto is_dead)
    {
        {
            auto l = sstl::with_guard(m_mu);
            m_tensors.erase(full_key);
        }
        done(s, send_args, recv_args, val, is_dead);
    };

    return BaseRemoteRendezvous::RecvAsync(key, args, std::move(final_done));
}

void WorkerRendezvous::RecvFromRemoteAsync(const ParsedKey &, const Args &recv_args, DoneCallback done)
{
    LOG(ERROR) << "Salus WorkerRendezvous only supports local recv";
    done(tf::errors::Internal("Salus WorkerRendezvous only supports local recv"), {}, recv_args, {}, false);
}

void WorkerRendezvous::SameWorkerRecvDone(const ParsedKey &parsed, const Args &send_args,
                                          const Args &recv_args, const tf::Tensor &in, tf::Tensor *out,
                                          tf::StatusCallback done)
{
#if defined(SALUS_ENABLE_SIEXECUTOR)
    auto &device_mgr = session()->device_mgr;

    tf::Device *send_dev = nullptr;
    tf::DeviceContext *send_dctx = send_args.device_context;
    auto s = device_mgr->LookupDevice(parsed.src_device, &send_dev);
    if (!s.ok()) {
        done(s);
        return;
    }

    tf::Device *recv_dev = nullptr;
    tf::DeviceContext *recv_dctx = recv_args.device_context;
    s = device_mgr->LookupDevice(parsed.dst_device, &recv_dev);
    if (!s.ok()) {
        done(s);
        return;
    }
#else
    auto send_wrapper = sstl::wrap_unref(static_cast<DeviceContextWithDevice *>(send_args.device_context));
    auto recv_wrapper = sstl::wrap_unref(static_cast<DeviceContextWithDevice *>(recv_args.device_context));

    auto send_dev = send_wrapper->device();
    auto send_dctx = send_wrapper->wrapped();
    auto recv_dev = recv_wrapper->device();
    auto recv_dctx = recv_wrapper->wrapped();
#endif // SALUS_ENABLE_SIEXECUTOR

    // Do a quick copy (sharing the underlying buffer) if both tensors
    // are on host memory.
    const bool src_host =
        (send_args.alloc_attrs.on_host() || send_dev->attributes().device_type() == tf::DEVICE_CPU);
    const bool dst_host =
        (recv_args.alloc_attrs.on_host() || recv_dev->attributes().device_type() == tf::DEVICE_CPU);
    if (src_host && dst_host) {
        *out = in;
        done(tf::Status::OK());
        return;
    }

    // This copy must involve a GPU. Hence, "in" must support DMA
    // (e.g., string tensors do not work on GPU).
    if (!tf::DMAHelper::CanUseDMA(&in)) {
        done(tf::errors::InvalidArgument("Non-DMA-safe ", tf::DataTypeString(in.dtype()),
                                         " tensor may not be copied from/to a GPU."));
        return;
    }

    auto attr = recv_args.alloc_attrs;
    attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() || recv_args.alloc_attrs.gpu_compatible());
    auto *out_allocator = recv_dev->GetAllocator(attr);
    tf::Tensor copy(out_allocator, in.dtype(), in.shape());
    *out = copy;

    // The following function takes care of cpu->gpu, gpu->cpu, gpu->gpu copies,
    // etc.
    VLOG(2) << "WorkerRendezvous::SameWorkerRecvDone copy from " << send_dev->name() << " to "
            << recv_dev->name() << "    send_on_host " << send_args.alloc_attrs.on_host() << " recv_on_host "
            << recv_args.alloc_attrs.on_host() << " src_data: " << reinterpret_cast<uint64_t>(in.tensor_data().data())
            << " dst_data: " << reinterpret_cast<uint64_t >(out->tensor_data().data());
    tf::CopyTensor::ViaDMA(parsed.edge_name, send_dctx, recv_dctx, send_dev, recv_dev, send_args.alloc_attrs,
                           attr, &in, out, done);
}

} // namespace salus::oplib::tensorflow
