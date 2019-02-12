/*
 * Copyright 2019 Peifeng Yu <peifeng@umich.edu>
 * 
 * This file is part of Salus
 * (see https://github.com/SymbioticLab/Salus).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    VLOG(2) << "MultiDeviceRendezvous::Send " << parsed;

    auto args = send_args;
    args.device_context = new DeviceContextWithDevice(m_device, sstl::add_ref(send_args.device_context));

    return m_local->Send(parsed, args, val, is_dead);
}

void RendezvousWithHook::RecvAsync(const ParsedKey &parsed, const Args &recv_args, DoneCallback done)
{
    VLOG(2) << "MultiDeviceRendezvous::RecvAsync " << parsed;

    auto args = recv_args;
    args.device_context = new DeviceContextWithDevice(m_device, sstl::add_ref(recv_args.device_context));

    m_local->RecvAsync(parsed, args, std::move(done));
}

void RendezvousWithHook::StartAbort(const tensorflow::Status &status)
{
    return m_local->StartAbort(status);
}
