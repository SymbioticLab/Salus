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

#include "platform/logging.h"
#include "utils/macros.h"
#include "utils/protoutils.h"
#include "utils/zmqutils.h"
#include "oplibraries/tensorflow/v2/md_executor.h"

#include "executor.pb.h"

#include <tensorflow/core/distributed_runtime/zrpc/exechelper/tfoplibraryproxy.h>
#include <tensorflow/core/protobuf/master.pb.h>
#include <tensorflow/core/protobuf/worker.pb.h>

#include <functional>
#include <unordered_map>

namespace zrpc = executor;

OpLibraryRegistary::Register tfoplibraryv2(executor::TENSORFLOW, std::make_unique<TFOpLibraryV2>(), 200);

TFOpLibraryV2::TFOpLibraryV2()
    : m_maxOpenSessions(0)
{
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
    using Method = std::function<void(ZmqServer::Sender&&, Proxy *,
                                      const zrpc::CustomRequest&, ITask::DoneCallback)>;

    static std::unordered_map<std::string, Method> funcs{
#define HANDLER(name) \
    {"tensorflow." #name "Request", \
        [this](auto sender, auto proxy, auto creq, auto cb) { \
            UNUSED(sender); \
            auto req = utils::createMessage<tensorflow::name##Request>("tensorflow." #name "Request", \
                                                                       creq.extra().data(), \
                                                                       creq.extra().size()); \
            if (!req) { \
                ERR("Failed to parse message"); \
                cb(nullptr); \
                return; \
            } \
            auto preq = req.release(); \
            proxy->Handle##name(preq, [cb, preq](auto resp, auto status){ \
                delete preq; \
                auto cresp = std::make_unique<zrpc::CustomResponse>(); \
                cresp->mutable_result()->set_code(status.code()); \
                cresp->mutable_result()->set_message(status.error_message()); \
                if (resp && status.ok()) { \
                    resp->SerializeToString(cresp->mutable_extra()); \
                } \
                delete resp; \
                cb(std::move(cresp)); \
            }); \
        } \
    },

        HANDLER(CreateSession)
        HANDLER(ExtendSession)
        HANDLER(PartialRunSetup)
        HANDLER(RunStep)
//         HANDLER(CloseSession)
        HANDLER(ListDevices)
        HANDLER(Reset)

        CallWithWorkerMethodName(HANDLER)
        HANDLER(RunGraph)
#undef HANDLER

        {"tensorflow.RecvTensorRequest", [this](auto sender, auto proxy, auto creq, auto cb) {
            auto req = utils::createMessage<tensorflow::RecvTensorRequest>("tensorflow.RecvTensorRequest",
                                                                           creq.extra().data(),
                                                                           creq.extra().size());
            if (!req) {
                ERR("Failed to parse message");
                cb(nullptr);
                return;
            }
            auto preq = req.release();
            proxy->HandleRecvTensorRaw(preq, [sender, cb, preq](auto resp, auto status) {
                UNUSED(status);
                sender->sendMessage("tensorflow.RecvTensorResponse", MultiPartMessage(resp));
                delete preq;
                delete resp;
                cb(nullptr);
            });
        }},
        {"tensorflow.CloseSessionRequest", [this, recvId](auto sender, auto proxy, auto creq, auto cb) {
            UNUSED(sender);
            this->handleCloseSession(recvId, proxy, creq, cb);
        }},
    };

    auto it = funcs.find(req.type());
    if (it == funcs.end()) {
        ERR("{} not found in registered custom tasks", req.type());
        return nullptr;
    }

    DEBUG("Dispatching custom task {} of seq {}", it->first, evenlop.seq());
    return make_async_lambda_task([this,
                                  recvId,
                                  sender = std::move(sender),
                                  req,
                                  fn = it->second](auto cb) mutable {
        auto proxy = getOrCreateProxy(recvId);
        fn(std::move(sender), proxy, req, cb);
    });
}

tensorflow::remote::TFOpLibraryProxy * TFOpLibraryV2::getOrCreateProxy(const std::string& recvId)
{
    std::lock_guard<std::mutex> g(m_mu);
    auto &p = m_proxies[recvId];
    if (!p) {
        INFO("Creating proxy object for recv id {}", recvId);
        p = std::make_unique<tensorflow::remote::TFOpLibraryProxy>(NewMultiDeviceExecutor);
        auto s = p->init();
        if (!s.ok()) {
            ERR("Failed to create a proxy object for recv id {}: {}", recvId, s);
            p.reset();
        }
    }
    if (m_proxies.size() > m_maxOpenSessions) {
        m_maxOpenSessions = m_proxies.size();
    }
    return p.get();
}

void TFOpLibraryV2::deregisterProxy(const std::string &recvId)
{
    std::lock_guard<std::mutex> g(m_mu);
    auto c = m_proxies.erase(recvId);
    if (c == 0) {
        WARN("No proxy object found to deregister for recvId {}", recvId);
    }
}

void TFOpLibraryV2::handleCloseSession(const std::string &recvId, Proxy *proxy, const executor::CustomRequest &creq, ITask::DoneCallback cb)
{
    auto req = utils::createMessage<tensorflow::CloseSessionRequest>("tensorflow.CloseSessionRequest", creq.extra().data(), creq.extra().size());
    if (!req) {
        ERR("Failed to parse message");
        cb(nullptr);
        return;
    }

    auto preq = req.release();
    proxy->HandleCloseSession(preq, [this, recvId, cb, preq](auto resp, auto status) {
        this->deregisterProxy(recvId);
        delete preq;

        auto cresp = std::make_unique<zrpc::CustomResponse>();
        cresp->mutable_result()->set_code(status.code());
        cresp->mutable_result()->set_message(status.error_message());
        if (resp && status.ok()) {
            resp->SerializeToString(cresp->mutable_extra());
        }
        delete resp;
        cb(std::move(cresp));
    });
}
