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

#include "executor.pb.h"

#include <tensorflow/core/distributed_runtime/zrpc/exechelper/tfoplibraryproxy.h>
#include <tensorflow/core/protobuf/master.pb.h>
#include <tensorflow/core/protobuf/worker.pb.h>

#include <functional>
#include <unordered_map>

namespace zrpc = executor;

OpLibraryRegistary::Register tfoplibraryv2(executor::TENSORFLOW, std::make_unique<TFOpLibraryV2>(), 200);

TFOpLibraryV2::TFOpLibraryV2()
    : m_proxy(std::make_unique<tensorflow::remote::TFOpLibraryProxy>())
    , m_proxyInitialized(false)
{
}

TFOpLibraryV2::~TFOpLibraryV2() = default;

bool TFOpLibraryV2::accepts(const zrpc::OpKernelDef &operation)
{
    return operation.oplibrary() == zrpc::TENSORFLOW;
}

bool TFOpLibraryV2::ensureProxyInitialized()
{
    std::lock_guard<std::mutex> g(m_mu);
    if (m_proxyInitialized) {
        return true;
    }
    auto s = m_proxy->init();
    if (s.ok()) {
        m_proxyInitialized = true;
        return true;
    }
    ERR("Error when initializing TFOpLibraryProxy: {}", s);
    return false;
}

#define INITIALIZE_PROXY_OR_RETURN(ret) \
    do { \
        if (!ensureProxyInitialized()) { \
            return ret; \
        } \
    } while (false)

PTask TFOpLibraryV2::createRunGraphTask(ZmqServer::Sender sender, const zrpc::EvenlopDef &evenlop,
                                        const zrpc::RunGraphRequest &request)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(request);

    INITIALIZE_PROXY_OR_RETURN({});

    return {};
}

PTask TFOpLibraryV2::createRunTask(ZmqServer::Sender sender, const zrpc::EvenlopDef &evenlop,
                                   const zrpc::RunRequest &request)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(request);

    INITIALIZE_PROXY_OR_RETURN({});

    return {};
}

PTask TFOpLibraryV2::createCustomTask(ZmqServer::Sender sender, const zrpc::EvenlopDef &evenlop,
                                      const zrpc::CustomRequest &req)
{
    // NOTE: this-> is need to workaround a bug in GCC 6.x where member function lookup is broken
    // for generic lambda. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61636
    using Method = std::function<void(ZmqServer::Sender&&, const zrpc::CustomRequest&, ITask::DoneCallback)>;

    static std::unordered_map<std::string, Method> funcs{
#define HANDLER(name) \
    {"tensorflow." #name "Request", \
        [this](auto sender, auto creq, auto cb) { \
            UNUSED(sender); \
            auto req = utils::createMessage<tensorflow::name##Request>("tensorflow." #name "Request", creq.extra().data(), creq.extra().size()).release(); \
            if (!req) { \
                ERR("Failed to parse message"); \
                cb(nullptr); \
            } \
            m_proxy->Handle##name(req, [cb, req](auto resp, auto status){ \
                delete req; \
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

        CallWithMasterMethodName(HANDLER)
        CallWithWorkerMethodName(HANDLER)
        HANDLER(RunStep)
        HANDLER(RunGraph)
#undef HANDLER

        {"tensorflow.RecvTensorRequest", [this](auto sender, auto creq, auto cb) {
            auto req = utils::createMessage<tensorflow::RecvTensorRequest>("tensorflow.RecvTensorRequest", creq.extra().data(), creq.extra().size()).release();
            m_proxy->HandleRecvTensorRaw(req, [sender, cb, req](auto resp, auto status) {
                UNUSED(status);
                sender->sendMessage("tensorflow.RecvTensorResponse", MultiPartMessage(resp));
                delete req;
                delete resp;
                cb(nullptr);
            });
        }},
    };

    INITIALIZE_PROXY_OR_RETURN({});

    auto it = funcs.find(req.type());
    if (it == funcs.end()) {
        ERR("{} not found in registered custom tasks", req.type());
        return nullptr;
    }

    DEBUG("Dispatching custom task {} of seq {}", it->first, evenlop.seq());
    return make_async_lambda_task([sender = std::move(sender), req, fn = it->second](auto cb) mutable {
        fn(std::move(sender), req, cb);
    });
}
