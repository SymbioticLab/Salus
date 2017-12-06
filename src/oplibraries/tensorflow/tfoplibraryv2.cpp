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
 */

/*
 * Make sure tensorflow_headers is included first before
 * any other headers, so we can correctly override TF logging
 * with ours.
 */
#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "tfoplibraryv2.h"

#include "execution/executionengine.h"
#include "execution/resources.h"
#include "oplibraries/tensorflow/v2/md_executor.h"
#include "oplibraries/tensorflow/v2/tfallocator.h"
#include "platform/logging.h"
#include "utils/macros.h"
#include "utils/protoutils.h"
#include "utils/zmqutils.h"

#include "protos.h"

#include <functional>
#include <unordered_map>

namespace zrpc = executor;
using namespace tensorflow::remote;

namespace {
template<typename RESPONSE>
std::unique_ptr<zrpc::CustomResponse> consumeResponse(RESPONSE *resp, tf::Status s)
{
    auto cresp = std::make_unique<zrpc::CustomResponse>();
    cresp->mutable_result()->set_code(s.code());
    cresp->mutable_result()->set_message(s.error_message());
    if (resp && s.ok()) {
        resp->SerializeToString(cresp->mutable_extra());
    }
    delete resp;

    return cresp;
}
} // namespace

OpLibraryRegistary::Register tfoplibraryv2(executor::TENSORFLOW, std::make_unique<TFOpLibraryV2>(), 200);

bool TFOpLibraryV2::initialize()
{
    m_proxy = std::make_unique<TFOpLibraryProxy>();
    tf::ConfigProto config;
    //     config.mutable_gpu_options()->set_allow_growth(true);
    //     config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.00001);
    auto s = m_proxy->globalInit(config);
    if (!s.ok()) {
        LOG(ERROR) << "Failed to initialize proxy object: " << s;
        return false;
    }
    return true;
}

void TFOpLibraryV2::uninitialize()
{
    VLOG(2) << "Max open sessions: " << m_maxOpenSessions;
    m_proxy.reset();
}

bool TFOpLibraryV2::accepts(const zrpc::OpKernelDef &operation)
{
    return operation.oplibrary() == zrpc::TENSORFLOW;
}

void TFOpLibraryV2::onRunGraph(ZmqServer::Sender sender, const zrpc::EvenlopDef &evenlop,
                               const zrpc::RunGraphRequest &request, DoneCallback cb)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(request);

    cb(nullptr);
}

void TFOpLibraryV2::onRun(ZmqServer::Sender sender, const zrpc::EvenlopDef &evenlop,
                          const zrpc::RunRequest &request, DoneCallback cb)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(request);

    cb(nullptr);
}

void TFOpLibraryV2::onCustom(ZmqServer::Sender sender, const zrpc::EvenlopDef &evenlop,
                             const zrpc::CustomRequest &req, DoneCallback cb)
{
    auto recvId = evenlop.recvidentity();

    // NOTE: this-> is need to workaround a bug in GCC 6.x where member function lookup is broken
    // for generic lambda. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61636
    using Method = std::function<void(ZmqServer::Sender &&, const std::string &, const zrpc::CustomRequest &,
                                      DoneCallback)>;

    static std::unordered_map<std::string, Method> funcs{
#define HANDLER(name, sessHandle)                                                                            \
    {"tensorflow." #name "Request", [this](auto sender, auto recvId, auto creq, auto cb) {                   \
         UNUSED(sender);                                                                                     \
         UNUSED(recvId);                                                                                     \
         auto req =                                                                                          \
             utils::createMessage<tensorflow::name##Request>("tensorflow." #name "Request",                  \
                                                             creq.extra().data(), creq.extra().size());      \
         if (!req) {                                                                                         \
             LOG(ERROR) << "Failed to parse message";                                                        \
             cb(nullptr);                                                                                    \
             return;                                                                                         \
         }                                                                                                   \
         auto proxy = getProxy(sessHandle);                                                                  \
         auto preq = req.release();                                                                          \
         proxy->Handle##name(preq, [cb, preq](auto resp, auto status) {                                      \
             delete preq;                                                                                    \
             cb(consumeResponse(resp, status));                                                              \
         });                                                                                                 \
     }},

        // Master handlers
        //         HANDLER(CreateSession)
        HANDLER(ExtendSession, req->session_handle()) HANDLER(PartialRunSetup, req->session_handle())
            HANDLER(RunStep, req->session_handle())
        //         HANDLER(CloseSession)
        HANDLER(ListDevices, sessionFromRecvId(recvId)) HANDLER(Reset, sessionFromRecvId(recvId))

    // Only local worker is used. No worker request handlers
#undef HANDLER

            {"tensorflow.CreateSessionRequest",
             [this](auto sender, auto recvId, auto creq, auto cb) {
                 UNUSED(sender);
                 this->handleCreateSession(recvId, creq, cb);
             }},
        {"tensorflow.CloseSessionRequest",
         [this](auto sender, auto recvId, auto creq, auto cb) {
             UNUSED(sender);
             this->handleCloseSession(recvId, creq, cb);
         }},
    };

    auto it = funcs.find(req.type());
    if (it == funcs.end()) {
        LOG(ERROR) << req.type() << " not found in registered custom tasks";
        cb(nullptr);
    }

    VLOG(2) << "Dispatching custom task " << it->first << " of seq " << evenlop.seq();
    it->second(std::move(sender), recvId, req, std::move(cb));
}

std::unique_ptr<TFOpLibraryV2::Proxy> TFOpLibraryV2::createProxy()
{
    std::unique_ptr<TFOpLibraryV2::Proxy> p;
    auto s = m_proxy->newSession(p);
    if (!s.ok()) {
        LOG(ERROR) << "Failed to create a proxy object: " << s;
    }
    return p;
}

TFOpLibraryV2::Proxy *TFOpLibraryV2::getProxy(const std::string &sessHandle)
{
    std::lock_guard<std::mutex> g(m_mu);
    auto &p = m_proxies[sessHandle].proxy;
    if (!p) {
        if (sessHandle.empty()) {
            // special case for a default session proxy to handle requests without session_handle
            p = createProxy();
        } else {
            LOG(ERROR) << "Failed to find a proxy object for session " << sessHandle;
        }
    }
    return p.get();
}

const std::string &TFOpLibraryV2::sessionFromRecvId(const std::string &recvId)
{
    std::lock_guard<std::mutex> g(m_mu);
    return m_lastSession[recvId];
}

TFOpLibraryV2::ProxyAndInserter TFOpLibraryV2::deregisterProxy(const std::string &recvId,
                                                               const std::string &sessHandle)
{
    std::lock_guard<std::mutex> g(m_mu);
    m_lastSession.erase(recvId);
    auto it = m_proxies.find(sessHandle);
    if (it == m_proxies.end()) {
        LOG(WARNING) << "No proxy object found to deregister for session " << sessHandle;
        return {};
    }
    auto p = std::move(it->second);
    m_proxies.erase(it);
    return p;
}

void TFOpLibraryV2::registerProxy(const std::string &recvId, const std::string &sessHandle,
                                  std::unique_ptr<TFOpLibraryV2::Proxy> &&proxy,
                                  ExecutionEngine::Inserter inserter)
{
    std::lock_guard<std::mutex> g(m_mu);
    m_lastSession[recvId] = sessHandle;
    auto & [p, ins] = m_proxies[sessHandle];
    if (p) {
        LOG(WARNING) << "Overwriting an existing proxy registered for session " << sessHandle << ": "
                     << as_hex(p);
        p.reset();
    }
    p = std::move(proxy);
    ins = std::move(inserter);
    if (m_proxies.size() > m_maxOpenSessions) {
        m_maxOpenSessions = m_proxies.size();
    }
}

void TFOpLibraryV2::handleCreateSession(const std::string &recvId, const executor::CustomRequest &creq,
                                        DoneCallback cb)
{
    auto req =
        utils::createMessage<tensorflow::CreateSessionRequest>("tensorflow.CreateSessionRequest",
                                                               creq.extra().data(), creq.extra().size());
    if (!req) {
        LOG(ERROR) << "Failed to parse message";
        cb(nullptr);
        return;
    }

    ResourceMap rm;

    auto &m = req->config().zmq_options().resource_map();
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

    uint64_t ticket;
    if (!SessionResourceTracker::instance().admit(rm, ticket)) {
        LOG(WARNING) << "Rejecting session due to unsafe resource usage. Predicted usage: "
                     << rm.DebugString()
                     << ", current usage: " << SessionResourceTracker::instance().DebugString();
        cb(consumeResponse<tf::CreateSessionResponse>(nullptr,
                                                      tf::errors::Internal("Session memory usage unsafe")));
        return;
    }

    auto proxy = createProxy();
    auto preq = req.release();
    proxy->HandleCreateSession(preq, [this, ticket, preq, recvId, cb, pproxy = proxy.release()](auto resp,
                                                                                                auto status) {
        std::unique_ptr<Proxy> proxy(pproxy);
        delete preq;
        if (status.ok()) {
            VLOG(2) << "Session " << resp->session_handle() << " created with recvId " << recvId;
            auto ins = ExecutionEngine::instance().registerSession(resp->session_handle());
            proxy->setExecFactory([ins](auto params, auto graph, auto executor) {
                return NewMultiDeviceExecutor(params, graph, ins, executor);
            });
            SessionResourceTracker::instance().acceptAdmission(ticket, resp->session_handle());
            registerProxy(recvId, resp->session_handle(), std::move(proxy), std::move(ins));
        } else {
            SessionResourceTracker::instance().free(ticket);
        }
        cb(consumeResponse(resp, status));
    });
}

void TFOpLibraryV2::handleCloseSession(const std::string &recvId, const executor::CustomRequest &creq,
                                       DoneCallback cb)
{
    auto req = utils::createMessage<tf::CloseSessionRequest>("tensorflow.CloseSessionRequest",
                                                             creq.extra().data(), creq.extra().size());
    if (!req) {
        LOG(ERROR) << "Failed to parse message";
        cb(nullptr);
        return;
    }

    auto[proxy, ins] = deregisterProxy(recvId, req->session_handle());
    if (!proxy) {
        cb(consumeResponse<tf::CloseSessionResponse>(nullptr, tf::errors::Internal(
                                                                  "No session object found to close")));
        return;
    }

    auto preq = req.release();
    auto pproxy = proxy.release();
    pproxy->HandleCloseSession(preq, [cb, preq, pproxy, ins = std::move(ins)](auto resp, auto status) {
        std::unique_ptr<tf::CloseSessionRequest> req(preq);
        ins->deleteSession([pproxy]() { std::unique_ptr<Proxy> proxy(pproxy); });
        SessionResourceTracker::instance().free(req->session_handle());
        cb(consumeResponse(resp, status));
    });
}
