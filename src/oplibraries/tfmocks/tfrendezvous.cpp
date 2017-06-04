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

TFRendezvous::Item::Item()
    : Item("", Args(), false, tensorflow::Tensor())
{ }

TFRendezvous::Item::Item(const std::string &k, const Args &a, bool d, tensorflow::Tensor &&v)
    : key(k)
    , args(a)
    , isDead(d)
    , val(std::move(v))
{ }

TFRendezvous::TFRendezvous(TFSession *sess)
    : m_sess(sess)
{ }

TFRendezvous::~TFRendezvous() { }

tensorflow::Status TFRendezvous::Send(const ParsedKey& parsed, const Args& send_args,
                                      const tensorflow::Tensor& val, const bool is_dead)
{
    UNUSED(parsed);
    UNUSED(send_args);
    UNUSED(is_dead);

    INFO("TFRendezvous::Send");
    m_sess->registerTensorMemory(val);

    tensorflow::Tensor copy(val);
    m_tensors.emplace_back(parsed.FullKey().ToString(), send_args, is_dead, std::move(copy));

    return tensorflow::Status::OK();
}

void TFRendezvous::RecvAsync(const ParsedKey& parsed, const Args& recv_args, DoneCallback done)
{
    UNUSED(parsed);
    UNUSED(recv_args);
    UNUSED(done);

    INFO("TFRendezvous::RecvAsync");
    // TODO: receive from tensorflow cpu
}

void TFRendezvous::StartAbort(const tensorflow::Status& status)
{
    UNUSED(status);
}

const TFRendezvous::TensorTable &TFRendezvous::receivedTensors() const
{
    return m_tensors;
}
