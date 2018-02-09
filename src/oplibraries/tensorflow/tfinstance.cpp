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
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/tfsession.h"

namespace symbiotic::salus::oplib::tensorflow {

/* static */ TFInstance &TFInstance::instance()
{
    static TFInstance inst;
    return inst;
}

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

void TFInstance::handleCreateSession(ZmqServer::Sender sender, const tf::CreateSessionRequest &req, tf::CreateSessionResponse &resp,
                                     StatusCallback &&cb)
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

    auto *gdef = const_cast<CreateSessionRequest &>(req)->mutable_graph_def();
    auto session = std::make_shared<TFSession>(*this, std::move(inserter), req.config(), gdef);

    resp.set_session_handle(session->handle());
    // Insert into the session map, which takes ownership of the session.
    {
        salus::Guard l(m_mu);
        DCHECK(m_sessions.try_emplace(session->handle(), std::move(session)).second);
    }

    cb(Status::OK());
}

std::shared_ptr<TFSession> TFInstance::findSession(const std::string &sessHandle)
{
    salus::Guard g(m_mu);
    auto it = m_sessions.find(sessHandle);
    if (it == m_sessions.end()) {
        throw TFException(tf::errors::InvalidArgument("Session ", sessHandle,
                                                      " is not found. Possibly, this master has restarted."));
    }
    return it->second;
}

std::shared_ptr<TFSession> TFInstance::popSession(const std::string &sessHandle)
{
    salus::Guard g(m_mu);
    auto nh = m_sessions.extract(sessHandle);
    if (!nh) {
        throw TFException(tf::errors::InvalidArgument("Session ", sessHandle,
                                                      " is not found. Possibly, this master has restarted."));
    }
    return std::move(nh.mapped());
}

void TFInstance::handleCloseSession(ZmqServer::Sender sender, const tf::CloseSessionRequest &req, tf::CloseSessionResponse &resp,
                                    StatusCallback &&cb)
{
    auto sess = popSession(req->session_handle());
    sess->safeClose();
    cb(Status::OK());
}

void TFInstance::handleListDevices(ZmqServer::Sender sender, const tf::ListDevicesRequest &req, tf::ListDevicesResponse &resp, StatusCallback &&cb)
{
    UNUSED(sender);
    UNUSED(req);
    for (auto dev : devices()) {
        *(resp->add_local_device()) = dev->attributes();
    }
    cb(Status::OK());
}

void TFInstance::handleReset(ZmqServer::Sender sender, const tf::ResetRequest &req, tf::ResetResponse &resp, StatusCallback &&cb)
{
    std::vector<std::shared_ptr<TFSession>> sessToClose;
    {
        salus::Guard g(m_mu);
        sessToClose.reserve(m_sessions.size());
        for (auto &[sessHandle, sess] : m_sessions) {
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

} // namespace symbiotic::salus::oplib::tensorflow
