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

/*
 * Make sure tensorflow_headers is included first before
 * any other headers, so we can correctly override TF logging
 * with ours.
 */
#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "md_rendezvous.h"

#include "utils/macros.h"

using tensorflow::Status;
using tensorflow::Tensor;

LocalWrapperRendezvous::LocalWrapperRendezvous(const std::shared_ptr<tensorflow::Device> &device,
                                             tensorflow::Rendezvous *rendez)
    : m_device(device)
    , m_local(rendez)
{
    m_local->Ref();
}

LocalWrapperRendezvous::~LocalWrapperRendezvous()
{
    m_local->Unref();
}

tensorflow::Status LocalWrapperRendezvous::Send(const ParsedKey &parsed, const Args &send_args,
                                      const tensorflow::Tensor &val, const bool is_dead)
{
    VLOG(2) << "MultiDeviceRendezvous::Send " << parsed.FullKey().ToString();

    auto args = send_args;
    args.device_context = new tf::WrapperDeviceContext(m_device, send_args.device_context);

    return m_local->Send(parsed, args, val, is_dead);
}

void LocalWrapperRendezvous::RecvAsync(const ParsedKey &parsed, const Args &recv_args, DoneCallback done)
{
    VLOG(2) << "MultiDeviceRendezvous::RecvAsync " << parsed.FullKey().ToString();

    auto args = recv_args;
    args.device_context = new tf::WrapperDeviceContext(m_device, recv_args.device_context);

    m_local->RecvAsync(parsed, args, std::move(done));
}

void LocalWrapperRendezvous::StartAbort(const tensorflow::Status &status)
{
    return m_local->StartAbort(status);
}
