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

#include <tensorflow/core/util/device_name_utils.h>

using tensorflow::Status;
using tensorflow::Tensor;

namespace {
bool isSameDevice(const tensorflow::DeviceNameUtils::ParsedName &a,
                  const tensorflow::DeviceNameUtils::ParsedName &b)
{
    return tensorflow::DeviceNameUtils::IsSameAddressSpace(a, b) &&
        (a.has_type && b.has_type && (a.type == b.type)) &&
        (a.has_id && b.has_id && (a.id == b.id));
}
} // namespace

TFRendezvous::SendItem::SendItem() : SendItem(Args(), false, tensorflow::Tensor()) {}

TFRendezvous::SendItem::SendItem(const Args &a, bool d, tensorflow::Tensor &&v)
    : args(a)
    , isDead(d)
    , val(std::move(v))
{
}

TFRendezvous::RecvItem::RecvItem() : RecvItem(Args()) {}

TFRendezvous::RecvItem::RecvItem(const Args &a) : args(a) {}

TFRendezvous::TFRendezvous(TFExecutionState *exec)
    : m_exec(exec)
    , m_local(m_exec->rendez())
{
    m_local->Ref();
}

TFRendezvous::~TFRendezvous()
{
    m_local->Unref();
}
tensorflow::Status TFRendezvous::triggerSend(const ParsedKey& parsed, const Args& send_args,
                                             const tensorflow::Tensor& val, const bool is_dead)
{
    return m_local->Send(parsed, send_args, val, is_dead);
}

tensorflow::Status TFRendezvous::Send(const ParsedKey &parsed, const Args &send_args,
                                      const tensorflow::Tensor &val, const bool is_dead)
{
    INFO("TFRendezvous::Send {}", parsed.FullKey().ToString());

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
            m_tensors.emplace(std::make_pair(parsed.FullKey().ToString(),
                                             SendItem {send_args, is_dead, std::move(copy)}));
        }
    }

    return m_local->Send(parsed, send_args, val, is_dead);
}

void TFRendezvous::RecvAsync(const ParsedKey &parsed, const Args &recv_args, DoneCallback done)
{
    INFO("TFRendezvous::RecvAsync {}", parsed.FullKey().ToString());

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
                done(tensorflow::errors::Internal("Duplicated recv"), Args(),
                     recv_args, tensorflow::Tensor(), false);
                return;
            }
            m_recv.emplace(std::make_pair(parsed.FullKey().ToString(), RecvItem {recv_args}));
        }
    }
    return m_local->RecvAsync(parsed, recv_args, std::move(done));
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
