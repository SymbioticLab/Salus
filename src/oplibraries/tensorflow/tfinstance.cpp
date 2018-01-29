/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Aetf <aetf@unlimitedcodeworks.xyz>
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

#include "tfinstance.h"

#include "oplibraries/tensorflow/tensorflow_headers.h"

namespace tf = ::tensorflow;

namespace symbiotic::salus::oplib::tensorflow {

TFInstance::TFInstance(const tf::ConfigProto &config)
    : m_env(tf::Env::Default())
{
    tf::SessionOptions sess_opts;
    // Disable old style RPCDevice creation if any.
    (*sess_opts.config.mutable_device_count())["RPC"] = 0;

    sess_opts.config.MergeFrom(config);

    // Load devices
    SALUS_THROW_IF_ERROR(tf::DeviceFactory::AddDevices(sess_opts, namePrefix(), &m_devices));
    m_deviceMgr = std::make_unique<tf::DeviceMgr>(m_devices);
}

TFInstance::~TFInstance() = default;

auto TFInstance::createSession()
{
    return std::make_unique<TFSession>(*this);
}

} // namespace symbiotic::salus::oplib::tf
