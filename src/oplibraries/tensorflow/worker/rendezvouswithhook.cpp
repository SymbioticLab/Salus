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
#include "oplibraries/tensorflow/worker/rendezvouswithhook.h"
#include "oplibraries/tensorflow/worker/devicecontextwithdevice.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "utils/macros.h"

using namespace salus::oplib::tensorflow;

RendezvousWithHook::RendezvousWithHook(std::shared_ptr<tensorflow::Device> device,
                                       sstl::ScopedUnref<tensorflow::Rendezvous> rendez)
    : m_device(std::move(device))
    , m_local(std::move(rendez))
{
}

RendezvousWithHook::~RendezvousWithHook() = default;

tensorflow::Status RendezvousWithHook::Send(const ParsedKey &parsed, const Args &send_args,
                                            const tensorflow::Tensor &val, const bool is_dead)
{
    VLOG(2) << "MultiDeviceRendezvous::Send " << parsed.FullKey().ToString();

    auto args = send_args;
    args.device_context = new DeviceContextWithDevice(m_device, sstl::add_ref(send_args.device_context));

    return m_local->Send(parsed, args, val, is_dead);
}

void RendezvousWithHook::RecvAsync(const ParsedKey &parsed, const Args &recv_args, DoneCallback done)
{
    VLOG(2) << "MultiDeviceRendezvous::RecvAsync " << parsed.FullKey().ToString();

    auto args = recv_args;
    args.device_context = new DeviceContextWithDevice(m_device, sstl::add_ref(recv_args.device_context));

    m_local->RecvAsync(parsed, args, std::move(done));
}

void RendezvousWithHook::StartAbort(const tensorflow::Status &status)
{
    return m_local->StartAbort(status);
}
