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

using tensorflow::Status;
using tensorflow::Tensor;

TFRendezvous::Item::Item()
    : Item(Args(), false, tensorflow::Tensor())
{
}

TFRendezvous::Item::Item(const Args &a, bool d, tensorflow::Tensor &&v)
    : args(a)
    , isDead(d)
    , val(std::move(v))
{
}

TFRendezvous::TFRendezvous(TFSession *sess)
    : m_sess(sess)
    , m_local(tensorflow::NewLocalRendezvous())
{
}

TFRendezvous::~TFRendezvous()
{
    m_local->Unref();
}

tensorflow::Status TFRendezvous::Send(const ParsedKey &parsed, const Args &send_args,
                                      const tensorflow::Tensor &val, const bool is_dead)
{
    INFO("TFRendezvous::Send {}", parsed.FullKey().ToString());

    auto key = parsed.FullKey().ToString();

    auto it = m_tensors.end();
    {
        tensorflow::mutex_lock locker(m_mu);
        it = m_tensors.find(key);
    }
    if (it == m_tensors.end()) {
        m_sess->registerTensorMemory(val);
        tensorflow::Tensor copy(val);
        {
            tensorflow::mutex_lock locker(m_mu);
            m_tensors.emplace(std::make_pair(parsed.FullKey().ToString(),
                                             Item {send_args, is_dead, std::move(copy)}));
        }
        return m_local->Send(parsed, send_args, val, is_dead);
    } else {
        ERR("Duplicated send: {}", key);
        return tensorflow::errors::Aborted("Duplicated send: ", parsed.FullKey());
    }
}

void TFRendezvous::RecvAsync(const ParsedKey &parsed, const Args &recv_args, DoneCallback done)
{
    INFO("TFRendezvous::RecvAsync {}", parsed.FullKey().ToString());
    // TODO: also handle receiving from tensorflow cpu

    return m_local->RecvAsync(parsed, recv_args,
                              [done, parsed, this](auto status, auto send_args, auto recv_args, auto val, auto is_dead){
        INFO("{} is received", parsed.FullKey().ToString());
        {
            tensorflow::mutex_lock locker(m_mu);
            m_tensors.erase(parsed.FullKey().ToString());
        }
        done(status, send_args, recv_args, val, is_dead);
    });
}

void TFRendezvous::StartAbort(const tensorflow::Status &status)
{
    return m_local->StartAbort(status);
}

TFRendezvous::TensorTable TFRendezvous::receivedTensors()
{
    TensorTable table;
    {
        tensorflow::mutex_lock locker(m_mu);
        std::swap(table, m_tensors);
    }
    return table;
}
