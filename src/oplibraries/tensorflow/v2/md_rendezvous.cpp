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

#include "md_rendezvous.h"

#include "platform/logging.h"
#include "utils/macros.h"

#include "oplibraries/tensorflow/tensorflow_headers.h"

using tensorflow::Status;
using tensorflow::Tensor;

MultiDeviceRendezvous::MultiDeviceRendezvous(tensorflow::Device *device,
                                             tf::WrapperDeviceContext::WrapperFunction allocWrapper,
                                             tensorflow::Rendezvous *rendez)
    : m_device(device)
    , m_allocWrapper(allocWrapper)
    , m_local(rendez)
{
    m_local->Ref();
}

MultiDeviceRendezvous::~MultiDeviceRendezvous()
{
    m_local->Unref();
}

tensorflow::Status MultiDeviceRendezvous::Send(const ParsedKey &parsed, const Args &send_args,
                                      const tensorflow::Tensor &val, const bool is_dead)
{
    INFO("MultiDeviceRendezvous::Send {}", parsed.FullKey().ToString());

    auto args = send_args;
    args.device_context =
        new tf::WrapperDeviceContext(static_cast<tensorflow::Device *>(m_device),
                                     m_allocWrapper,
                                     send_args.device_context);

    return m_local->Send(parsed, args, val, is_dead);
}

void MultiDeviceRendezvous::RecvAsync(const ParsedKey &parsed, const Args &recv_args, DoneCallback done)
{
    INFO("MultiDeviceRendezvous::RecvAsync {}", parsed.FullKey().ToString());

    auto args = recv_args;
    args.device_context =
        new tf::WrapperDeviceContext(static_cast<tensorflow::Device *>(m_device),
                                     m_allocWrapper,
                                     recv_args.device_context);

    m_local->RecvAsync(parsed, args, std::move(done));
}

void MultiDeviceRendezvous::StartAbort(const tensorflow::Status &status)
{
    return m_local->StartAbort(status);
}
