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
struct HandlerCallback
{
    DoneCallback cb;
    ProtoPtr tfresp;
    void operator ()(const Status &s) {
        auto cresp = std::make_unique<zrpc::CustomResponse>();
        cresp->mutable_result()->set_code(s.code());
        cresp->mutable_result()->set_message(s.error_message());
        if (tfresp && s.ok()) {
            tfresp->SerializeToString(cresp->mutable_extra());
        }
        cb(std::move(cresp));
    }
};

template<typename REQUEST>
std::pair<std::unique_ptr<REQUEST>, HandlerCallback> prepareTFCall(const zrpc::CustomRequest &creq);

#define IMPL_PARSE(name) \
template<> \
std::pair<std::unique_ptr<tf:: ## name ## Request>, ProtoPtr> \
prepareTFCall<tf:: ## name ## Request>(const zrpc::CustomRequest &creq) \
{ \
    auto tfreq = salus::createMessage<tf::## name ## Request>("tensorflow." #name "Request", \
                                                              creq.extra().data(), creq.extra().size()); \
    if (!tfreq) { \
        throw TFException(tf::errors("Failed to parse message as", "tensorflow." #name "Request")); \
    } \
\
    return std::make_pair(std::move(tfreq), std::make_unique<tf:: ## name ## Response>()); \
}

    CallWithMasterMethodName(IMPL_PARSE)

#undef IMPL_PARSE

OpLibraryRegistary::Register tfoplibraryv2(executor::TENSORFLOW, std::make_unique<TFOpLibraryV2>(), 200);

} // namespace

TFOpLibraryV2::~TFOpLibraryV2() = default;

bool TFOpLibraryV2::initialize()
{
    return true;
}

void TFOpLibraryV2::uninitialize()
{
    VLOG(2) << "Max open sessions: " << m_maxOpenSessions;
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
                             const zrpc::CustomRequest &creq, DoneCallback cb)
{
    using Method = std::function<void(ZmqServer::Sender &&, const zrpc::CustomRequest &, HandlerCallback&&)>;
    static std::unordered_map<std::string, Method> funcs{
#define INSTANCE_HANDLER(name) \
        {"tensorflow." #name "Request", [](auto sender, auto creq, auto &&hcb) { \
            auto [tfreq, hcb.tfresp] = prepareTFCall<tf::name ## Request>(creq); \
            TFInstance::instance().handle ## name (std::move(sender), *tfreq, *hcb.tfresp, std::move(hcb)); \
        }}

        INSTANCE_HANDLER(CreateSession),
        INSTANCE_HANDLER(CloseSession),
        INSTANCE_HANDLER(ListDevices),
        INSTANCE_HANDLER(Reset),

#undef INSTANCE_HANDLER

#define SESSION_HANDLER(name)                                                                            \
    {"tensorflow." #name "Request", [](auto sender, auto creq, auto &&hcb) {                   \
        auto [tfreq, hcb.tfresp] = prepareTFCall<tf::name ## Request>(creq); \
         auto sess = TFInstance::instance().findSession(creq->session_handle());                              \
         sess->handle##name(std::move(sender), *tfreq, *hcb.tfreqp, std::move(hcb)); \
     }}

        SESSION_HANDLER(ExtendSession),
        SESSION_HANDLER(PartialRunSetup),
        SESSION_HANDLER(RunStep),
#undef SESSION_HANDLER
    };

    HandlerCallback hcb{std::move(cb)};
    try {
        auto it = funcs.find(creq.type());
        if (it == funcs.end()) {
            throw TFException(tf::errors::InvalidArgument(creq.type(), " not found in registered custom tasks"));
        }

        VLOG(2) << "Dispatching custom task " << it->first << " of seq " << evenlop.seq();
        it->second(std::move(sender), creq, std::move(hcb));
    } catch (const TFException &ex) {
        LOG(ERROR) << ex.what();
        hcb(ex.code());
    }
}

void TFOpLibraryV2::handleCloseSession(const std::string &recvId, const executor::CustomRequest &creq,
                                       DoneCallback cb)
{
    auto req = salus::createMessage<tf::CloseSessionRequest>("tensorflow.CloseSessionRequest",
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
