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

#include "tfoplibraryv2.h"

#include "oplibraries/tensorflow/v2/md_executor.h"
#include "oplibraries/tensorflow/v2/tfallocator.h"
#include "platform/logging.h"
#include "utils/macros.h"
#include "utils/protoutils.h"
#include "utils/zmqutils.h"

#include "protos.h"

#include "oplibraries/tensorflow/tensorflow_headers.h"

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

TFOpLibraryV2::TFOpLibraryV2()
    : m_proxy(std::make_unique<TFOpLibraryProxy>(NewMultiDeviceExecutor))
    , m_maxOpenSessions(0)
{
    // use device with our own allocator
    WrappedDeviceSettings::maybeRegisterWrappedDeviceFactories();
    WrappedDeviceSettings::setWrapperFactory(
        [](auto *alloc, auto *) {
            if (alloc->Name() == "GPU_0_bfc") {
                // FIXME: there is memory leak
                alloc = new TrivialGPUAllocator(0);
            }
            return std::make_unique<TFAllocator>(alloc);
        });

    // Initialize proxy after set wrapper, as devices are created now.
    tf::ConfigProto config;
    config.mutable_gpu_options()->set_allow_growth(true);
    config.mutable_device_count()->set_per_process_gpu_memory_fraction(0.00001);
    auto s = m_proxy->globalInit(config);
    if (!s.ok()) {
        ERR("Failed to initialize proxy object: {}", s);
    }
}

TFOpLibraryV2::~TFOpLibraryV2()
{
    INFO("Max open sessions: {}", m_maxOpenSessions);
}

bool TFOpLibraryV2::accepts(const zrpc::OpKernelDef &operation)
{
    return operation.oplibrary() == zrpc::TENSORFLOW;
}

PTask TFOpLibraryV2::createRunGraphTask(ZmqServer::Sender sender, const zrpc::EvenlopDef &evenlop,
                                        const zrpc::RunGraphRequest &request)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(request);

    return {};
}

PTask TFOpLibraryV2::createRunTask(ZmqServer::Sender sender, const zrpc::EvenlopDef &evenlop,
                                   const zrpc::RunRequest &request)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(request);

    return {};
}

PTask TFOpLibraryV2::createCustomTask(ZmqServer::Sender sender, const zrpc::EvenlopDef &evenlop,
                                      const zrpc::CustomRequest &req)
{
    auto recvId = evenlop.recvidentity();

    // NOTE: this-> is need to workaround a bug in GCC 6.x where member function lookup is broken
    // for generic lambda. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61636
    using Method = std::function<void(ZmqServer::Sender &&, const std::string &, const zrpc::CustomRequest &,
                                      ITask::DoneCallback)>;

    static std::unordered_map<std::string, Method> funcs{
#define HANDLER(name, sessHandle)                                                                            \
    {"tensorflow." #name "Request", [this](auto sender, auto recvId, auto creq, auto cb) {                   \
         UNUSED(sender);                                                                                     \
         UNUSED(recvId);                                                                                     \
         auto req =                                                                                          \
             utils::createMessage<tensorflow::name##Request>("tensorflow." #name "Request",                  \
                                                             creq.extra().data(), creq.extra().size());      \
         if (!req) {                                                                                         \
             ERR("Failed to parse message");                                                                 \
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
        HANDLER(ExtendSession, req->session_handle())
        HANDLER(PartialRunSetup, req->session_handle())
        HANDLER(RunStep, req->session_handle())
//         HANDLER(CloseSession)
        HANDLER(ListDevices, sessionFromRecvId(recvId))
        HANDLER(Reset, sessionFromRecvId(recvId))

        // Worker handlers
        HANDLER(RegisterGraph, req->session_handle())
        HANDLER(RunGraph, sessionFromRecvId(recvId))
        HANDLER(GetStatus, sessionFromRecvId(recvId))
        HANDLER(DeregisterGraph, sessionFromRecvId(recvId))
        HANDLER(CleanupGraph, sessionFromRecvId(recvId))
        HANDLER(CleanupAll, sessionFromRecvId(recvId))
        HANDLER(Logging, sessionFromRecvId(recvId))
        HANDLER(Tracing, sessionFromRecvId(recvId))
#undef HANDLER

        {"tensorflow.RecvTensorRequest", [this](auto sender, auto recvId, auto creq, auto cb) {
            auto req = utils::createMessage<tensorflow::RecvTensorRequest>("tensorflow.RecvTensorRequest",
                                                                           creq.extra().data(),
                                                                           creq.extra().size());
            if (!req) {
                ERR("Failed to parse message");
                cb(nullptr);
                return;
            }
            auto proxy = getProxy(sessionFromRecvId(recvId));
            auto preq = req.release();
            proxy->HandleRecvTensorRaw(preq, [sender, cb, preq](auto resp, auto status) {
                UNUSED(status);
                sender->sendMessage("tensorflow.RecvTensorResponse", MultiPartMessage(resp));
                delete preq;
                delete resp;
                cb(nullptr);
            });
        }},
        {"tensorflow.CreateSessionRequest", [this](auto sender, auto recvId, auto creq, auto cb) {
            UNUSED(sender);
            this->handleCreateSession(recvId, creq, cb);
        }},
        {"tensorflow.CloseSessionRequest", [this](auto sender, auto recvId, auto creq, auto cb) {
            UNUSED(sender);
            this->handleCloseSession(recvId, creq, cb);
        }},
    };

    auto it = funcs.find(req.type());
    if (it == funcs.end()) {
        ERR("{} not found in registered custom tasks", req.type());
        return nullptr;
    }

    DEBUG("Dispatching custom task {} of seq {}", it->first, evenlop.seq());
    return make_async_lambda_task([sender = std::move(sender), recvId, req, it](auto cb) mutable {
        it->second(std::move(sender), recvId, req, cb);
    });
}

std::unique_ptr<TFOpLibraryV2::Proxy> TFOpLibraryV2::createProxy()
{
    std::unique_ptr<TFOpLibraryV2::Proxy> p;
    auto s = m_proxy->newSession(p);
    if (!s.ok()) {
        ERR("Failed to create a proxy object: {}", s);
    }
    return p;
}

TFOpLibraryV2::Proxy *TFOpLibraryV2::getProxy(const std::string &sessHandle)
{
    std::lock_guard<std::mutex> g(m_mu);
    auto &p = m_proxies[sessHandle];
    if (!p) {
        if (sessHandle.empty()) {
            // special case for a default session proxy to handle requests without session_handle
            p = createProxy();
        } else {
            ERR("Failed to find a proxy object for session {}", sessHandle);
        }
    }
    return p.get();
}

const std::string &TFOpLibraryV2::sessionFromRecvId(const std::string &recvId)
{
    std::lock_guard<std::mutex> g(m_mu);
    return m_lastSession[recvId];
}

std::unique_ptr<TFOpLibraryV2::Proxy> TFOpLibraryV2::deregisterProxy(const std::string &recvId,
                                                                     const std::string &sessHandle)
{
    std::lock_guard<std::mutex> g(m_mu);
    m_lastSession.erase(recvId);
    auto it = m_proxies.find(sessHandle);
    if (it == m_proxies.end()) {
        WARN("No proxy object found to deregister for session {}", sessHandle);
        return nullptr;
    }
    auto p = std::move(it->second);
    m_proxies.erase(it);
    return p;
}

void TFOpLibraryV2::registerProxy(const std::string &recvId, const std::string &sessHandle,
                                  std::unique_ptr<TFOpLibraryV2::Proxy> &&proxy)
{
    std::lock_guard<std::mutex> g(m_mu);
    m_lastSession[recvId] = recvId;
    auto &p = m_proxies[sessHandle];
    if (p) {
        WARN("Overwriting an existing proxy registered for session {}: {}", sessHandle, as_hex(p));
        p.reset();
    }
    p = std::move(proxy);
    if (m_proxies.size() > m_maxOpenSessions) {
        m_maxOpenSessions = m_proxies.size();
    }
}

void TFOpLibraryV2::handleCreateSession(const std::string &recvId, const executor::CustomRequest &creq,
                                        ITask::DoneCallback cb)
{
    auto req =
        utils::createMessage<tensorflow::CreateSessionRequest>("tensorflow.CreateSessionRequest",
                                                               creq.extra().data(), creq.extra().size());
    if (!req) {
        ERR("Failed to parse message");
        cb(nullptr);
        return;
    }

    INFO("Creating proxy object for recv id {}", recvId);
    auto proxy = createProxy();
    auto preq = req.release();
    proxy->HandleCreateSession(preq,
                               [this, preq, recvId, cb, pproxy = proxy.release()](auto resp, auto status) {
                                   std::unique_ptr<Proxy> proxy(pproxy);
                                   delete preq;
                                   if (status.ok()) {
                                       registerProxy(recvId, resp->session_handle(), std::move(proxy));
                                   }
                                   cb(consumeResponse(resp, status));
                               });
}

void TFOpLibraryV2::handleCloseSession(const std::string &recvId, const executor::CustomRequest &creq,
                                       ITask::DoneCallback cb)
{
    auto req = utils::createMessage<tf::CloseSessionRequest>("tensorflow.CloseSessionRequest",
                                                             creq.extra().data(), creq.extra().size());
    if (!req) {
        ERR("Failed to parse message");
        cb(nullptr);
        return;
    }

    auto proxy = deregisterProxy(recvId, req->session_handle());
    if (!proxy) {
        cb(consumeResponse<tf::CloseSessionResponse>(nullptr, tf::errors::Internal(
                                                                  "No session object found to close")));
        return;
    }

    auto preq = req.release();
    auto pproxy = proxy.release();
    pproxy->HandleCloseSession(preq, [this, cb, preq, pproxy](auto resp, auto status) {
        delete preq;
        delete pproxy;
        cb(consumeResponse(resp, status));
    });
}
