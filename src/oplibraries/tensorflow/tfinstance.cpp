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

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "tfinstance.h"
#include "execution/executionengine.h"
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/tfsession.h"
#include "utils/macros.h"

#if defined(WITH_GPERFTOOLS)
#include <gperftools/profiler.h>
#endif

namespace salus::oplib::tensorflow {

/* static */ TFInstance &TFInstance::instance()
{
    static TFInstance inst;
    return inst;
}

TFInstance::TFInstance()
    : m_env(tf::Env::Default())
{
    tf::SessionOptions sess_opts;
    // Disable old style RPCDevice creation if any.
    (*sess_opts.config.mutable_device_count())["RPC"] = 0;

    // Load devices
    maybeRegisterSalusDeviceFactories();
    SALUS_THROW_IF_ERROR(tf::DeviceFactory::AddDevices(sess_opts, namePrefix(), &m_devices));
    m_deviceMgr = std::make_unique<tf::DeviceMgr>(m_devices);

#if defined(WITH_GPERFTOOLS)
    ProfilerStart("/tmp/gperf..output");
#endif
}

TFInstance::~TFInstance()
{
#if defined(WITH_GPERFTOOLS)
    ProfilerStop();
#endif
}

void TFInstance::handleCreateSession(const tf::CreateSessionRequest &req, tf::CreateSessionResponse &resp,
                                     HandlerCallback &&cb)
{
    // Check session resource
    ResourceMap rm;

    auto &m = req.config().zmq_options().resource_map();
    for (auto p : m.persistant()) {
        auto tag = ResourceTag::fromString(p.first);
        if (tag.type == ResourceType::UNKNOWN) {
            continue;
        }
        rm.persistant[tag] = p.second;
    }

    for (auto p : m.temporary()) {
        auto tag = ResourceTag::fromString(p.first);
        if (tag.type == ResourceType::UNKNOWN) {
            continue;
        }
        rm.temporary[tag] = p.second;
    }

    auto inserter = ExecutionEngine::instance().createSessionOffer(rm);
    if (!inserter) {
        LOG(WARNING) << "Rejecting session due to unsafe resource usage. Predicted usage: "
                     << rm.DebugString()
                     << ", current usage: " << SessionResourceTracker::instance().DebugString();
        throw TFException(tf::errors::Internal("Session memory usage unsafe"));
    }

    SALUS_THROW_IF_ERROR(ValidateExternalGraphDefSyntax(req.graph_def()));

    auto *gdef = const_cast<tf::CreateSessionRequest &>(req).mutable_graph_def();
    auto session = std::make_shared<TFSession>(*this, std::move(inserter), req.config(), gdef);

    auto handle = session->handle();
    resp.set_session_handle(handle);
    // Insert into the session map, which takes ownership of the session.
    {
        sstl::Guard l(m_mu);
        if (!m_sessions.try_emplace(handle, std::move(session)).second) {
            throw TFException(tf::errors::Internal("Error when inserting session ", handle));
        }
    }
    LOG(INFO) << "Accepting and created session " << handle;

    cb(Status::OK());
}

std::shared_ptr<TFSession> TFInstance::findSession(const std::string &sessHandle)
{
    sstl::Guard g(m_mu);
    auto it = m_sessions.find(sessHandle);
    if (it == m_sessions.end()) {
        LOG(ERROR) << "Dumping all known sessions: ";
        for (auto & [h, ptr] : m_sessions) {
            LOG(ERROR) << " Session " << h << "@" << as_hex(ptr);
        }
        throw TFException(tf::errors::InvalidArgument("Session ", sessHandle,
                                                      " is not found. Possibly, this master has restarted."));
    }
    return it->second;
}

std::shared_ptr<TFSession> TFInstance::popSession(const std::string &sessHandle)
{
    sstl::Guard g(m_mu);
    auto nh = m_sessions.extract(sessHandle);
    if (!nh) {
        throw TFException(tf::errors::InvalidArgument("Session ", sessHandle,
                                                      " is not found. Possibly, this master has restarted."));
    }
    return std::move(nh.mapped());
}

void TFInstance::handleCloseSession(const tf::CloseSessionRequest &req, tf::CloseSessionResponse &resp,
                                    HandlerCallback &&cb)
{
    UNUSED(resp);
    auto sess = popSession(req.session_handle());
    LOG(INFO) << "Closing session " << sess->handle();
    sess->safeClose();
    cb(Status::OK());
}

void TFInstance::handleListDevices(const tf::ListDevicesRequest &req, tf::ListDevicesResponse &resp,
                                   HandlerCallback &&cb)
{
    UNUSED(req);
    for (auto dev : devices()) {
        *(resp.add_local_device()) = dev->attributes();
    }
    cb(Status::OK());
}

void TFInstance::handleReset(const tf::ResetRequest &req, tf::ResetResponse &resp, HandlerCallback &&cb)
{
    UNUSED(req);
    UNUSED(resp);
    std::vector<std::shared_ptr<TFSession>> sessToClose;
    {
        sstl::Guard g(m_mu);
        sessToClose.reserve(m_sessions.size());
        for (auto & [sessHandle, sess] : m_sessions) {
            UNUSED(sessHandle);
            sessToClose.emplace_back(std::move(sess));
        }
        m_sessions.clear();
    }

    Status s;
    for (auto &sess : sessToClose) {
        try {
            sess->safeClose();
        } catch (const TFException &ex) {
            s.Update(ex.code());
        }
    }
    sessToClose.clear();
    cb(s);
}

} // namespace salus::oplib::tensorflow
