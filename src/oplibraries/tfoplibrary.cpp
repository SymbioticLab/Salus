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

// Force to use non-debug version of protobuf map, which changes its hashing function
// according to debug state, causing problems when two libraries both use protobuf, but
// only one of them is built with debug. Then passing a map from one library to the other
// becomes impossible because values inserted using one hashing function can't be found
// using another hashing function.
#ifdef NDEBUG
#undef NDEBUG
#include "google/protobuf/map.h"
#define NDEBUG
#else
#include "google/protobuf/map.h"
#endif

#include "tfoplibrary.h"

#include "tfmocks/tfsession.h"
#include "tfmocks/tfallocator.h"
#include "tfmocks/tfrendezvous.h"

#include "utils/protoutils.h"
#include "utils/pointerutils.h"
#include "platform/logging.h"
#include "utils/macros.h"

#include "executor.pb.h"
#include "tfoplibrary.pb.h"

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_segment.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/function.pb.h>
#include <tensorflow/core/protobuf/config.pb.h>
#include <tensorflow/core/platform/mutex.h>
#include <tensorflow/core/lib/strings/strcat.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/common_runtime/device_factory.h>
#include <tensorflow/core/common_runtime/rpc_device/exec_helpers/devicefactories.h>

#include <vector>
#include <thread>
#include <functional>

namespace rpc = executor;
using namespace tensorflow::remote;
using ::tensorflow::NodeDef;
using ::tensorflow::ConfigProto;
using ::tensorflow::FunctionDefLibrary;
using ::google::protobuf::Message;
using std::unique_ptr;
using std::shared_ptr;

namespace {

void dumpOpKernel(tensorflow::OpKernel *opkernel)
{
    if (!opkernel) return;

    TRACE("m_opkernel.name() {}", opkernel->name());
    TRACE("m_opkernel.type_string() {}", opkernel->type_string());
    TRACE("m_opkernel.is_internal() {}", opkernel->is_internal());
    TRACE("m_opkernel.IsExpensive() {}", opkernel->IsExpensive());
    TRACE("m_opkernel.num_inputs() {}", opkernel->num_inputs());
    for (int i = 0; i != opkernel->num_inputs(); i++) {
        TRACE("m_opkernel.input_type({}) {}", i, opkernel->input_type(i));
    }
    for (size_t i = 0; i != opkernel->input_memory_types().size(); i++) {
        TRACE("m_opkernel.input_memory_types()[{}] {}", i, opkernel->input_memory_types()[i]);
    }

    TRACE("m_opkernel.num_outputs() {}", opkernel->num_outputs());
    for (int i = 0; i != opkernel->num_outputs(); i++) {
        TRACE("m_opkernel.output_type({}) {}", i, opkernel->output_type(i));
    }
    for (size_t i = 0; i != opkernel->output_memory_types().size(); i++) {
        TRACE("m_opkernel.output_memory_types()[{}] {}", i, opkernel->output_memory_types()[i]);
    }
    TRACE("m_opkernel.def() {}", opkernel->def());
}

void dumpOpContext(tensorflow::OpKernelContext *ctx)
{
    if (!ctx) return;

    for (int i = 0; i != ctx->num_inputs(); ++i) {
        TRACE("context.input_alloc_attr({}) {}", i, ctx->input_alloc_attr(i));
    }
    TRACE("context.is_output_dead() {}", *ctx->is_output_dead());
    TRACE("context.num_outputs() {}", ctx->num_outputs());
    for (int i = 0; i != ctx->num_outputs(); ++i) {
        TRACE("context.output_alloc_attr({}) {}", i, ctx->output_alloc_attr(i));
    }
}

} // namespace

REGISTER_OPLIBRARY(executor::TENSORFLOW, TFOpLibrary);

TFOpLibrary::~TFOpLibrary() = default;

bool TFOpLibrary::accepts(const rpc::OpKernelDef& operation)
{
    return operation.oplibrary() == rpc::TENSORFLOW;
}

TFSession *TFOpLibrary::getOrCreateSession(const std::string& sess_id,
                                           const tensorflow::ConfigProto& cfgProto)
{
    std::lock_guard<std::mutex> guard(m_mu);

    if (!m_deviceMgr) {
        std::vector<tensorflow::Device*> devices;
        tensorflow::SessionOptions options;
        (*options.config.mutable_device_count())["RPC"] = 0;

        // use device with our own allocator
        WrappedDeviceSettings::maybeRegisterWrappedDeviceFactories();
        WrappedDeviceSettings::setWrapperFactory([](auto *alloc){
            return std::make_unique<TFAllocator>(alloc);
        });

        auto s = tensorflow::DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0",
                                                       &devices);
        if (!s.ok()) {
            ERR("Error when adding devices to TFSession: {}", s);
            return nullptr;
        }
        m_deviceMgr = std::make_unique<tensorflow::DeviceMgr>(devices);
    }

    auto &sess = m_sessions[sess_id];
    if (!sess) {
        sess = std::make_unique<TFSession>(this, sess_id, cfgProto);
        if (!sess->initialize()) {
            sess.reset();
        }
    } else {
        DEBUG("Reuse previously created session");
    }

    return sess.get();
}

TFSession *TFOpLibrary::findSession(const std::string& sess_id)
{
    std::lock_guard<std::mutex> guard(m_mu);
    auto it = m_sessions.find(sess_id);
    if (it != m_sessions.end()) {
        return it->second.get();
    }
    ERR("Session {} not found", sess_id);
    return {};
}

void TFOpLibrary::destorySession(const std::string &sess_id)
{
    std::lock_guard<std::mutex> guard(m_mu);
    auto it = m_sessions.find(sess_id);
    if (it != m_sessions.end()) {
        m_sessions.erase(it);
    } else {
        WARN("Trying to destory nonexist session: {}", sess_id);
    }
}

PTask TFOpLibrary::createRunTask(ZmqServer::Sender sender, const rpc::EvenlopDef &evenlop,
                                 const rpc::RunRequest &request)
{
    auto tfdef = utils::createMessage<executor::TFOpKernelDef>("executor.TFOpKernelDef",
                                                             request.opkernel().extra().data(),
                                                             request.opkernel().extra().size());
    auto tfctxdef = utils::createMessage<executor::TFOpContextDef>("executor.TFOpContextDef",
                                                                   request.context().extra().data(),
                                                                   request.context().extra().size());

    if (!tfdef || !tfctxdef) { return {}; }

    auto sess = findSession(evenlop.sessionid());
    if (!sess) {
        return {};
    }

    auto exec = sess->findExecution(request.execid());
    if (!exec) {
        ERR("Execution {} not found in session {}", request.execid(), evenlop.sessionid());
        return {};
    }

    auto ndef = std::unique_ptr<NodeDef>(tfdef->release_nodedef());
    return std::make_unique<TFRunTask>(exec, std::move(sender), std::move(ndef),
                                       tfdef->isasync(), std::move(tfctxdef));
}

TFRunTask::TFRunTask(TFExecutionState *execState, ZmqServer::Sender &&sender, unique_ptr<NodeDef> &&nodedef,
                     bool async, unique_ptr<executor::TFOpContextDef> &&tfctxdef)
    : m_exec(execState)
    , m_sender(std::move(sender))
    , m_async(async)
    , m_ndef(std::move(nodedef))
    , m_tfctxdef(std::move(tfctxdef))
{
    assert(m_sender);
}

bool TFRunTask::isAsync()
{
    return m_async;
}

bool TFRunTask::prepare(DeviceSpec &dev)
{
    assert(m_sender);

    auto seq = m_sender->sequenceNumber();
    auto session = m_exec->session();

    tensorflow::OpKernel *kernel = nullptr;
    auto ok = session->findOrCreateKernel(m_exec, *m_ndef, kernel, dev);
    if (!ok) {
        if (!kernel) {
            ERR("Kernel creation failed");
        }
        // kernel on mismatched device
        return false;
    }
    if (m_async) {
        m_opkernel = kernel->AsAsync();
    } else {
        m_opkernel = kernel;
    }
    INFO("Created OpKernel for seq {}", seq);
    dumpOpKernel(m_opkernel);

    m_context = session->createContext(*m_tfctxdef, m_opkernel, seq, m_exec, dev);
    if (!m_context) {
        return false;
    }
    TRACE("Created OpKernelContext for seq {}", seq);
    dumpOpContext(m_context->ctx());

    m_context->devSpec = dev;

    return true;
}

ProtoPtr TFRunTask::run()
{
    INFO("running in thread {}", std::this_thread::get_id());
    auto resp = std::make_unique<executor::RunResponse>();

    if (!m_opkernel || !m_context) {
        ERR("Got nullptr for opkernel or context: m_opkernel = {:x}, m_context = {:x}",
            reinterpret_cast<uint64_t>(m_opkernel), reinterpret_cast<uint64_t>(m_context.get()));
        resp->mutable_result()->set_code(-1);
        return resp;
    }

    auto device = static_cast<tensorflow::Device*>(m_context->ctx()->device());
    if (!device) {
        ERR("Got nullptr for device in context");
        resp->mutable_result()->set_code(-1);
        return resp;
    }

    INFO("Running on device: {}", device->name());
    try {
        device->Compute(m_opkernel, m_context->ctx());
//         m_opkernel->Compute(m_context->ctx());
    } catch (std::exception &err) {
        ERR("Caught exception when run kernel compute: ", err.what());
    }

    INFO("OpKernel->Compute finished with status {}", m_context->ctx()->status());

    auto tfupd = m_exec->session()->finalizeContext(m_context.get());
    tfupd.SerializeToString(resp->mutable_context()->mutable_extra());

    return resp;
}

void TFRunTask::runAsync(DoneCallback cb)
{
    INFO("running async in thread {}", std::this_thread::get_id());

    if (!m_opkernel || !m_context) {
        ERR("Got nullptr for opkernel or context: m_opkernel = {:x}, m_context = {:x}",
            reinterpret_cast<uint64_t>(m_opkernel), reinterpret_cast<uint64_t>(m_context.get()));
        auto resp = std::make_unique<executor::RunResponse>();
        resp->mutable_result()->set_code(-1);
        cb(std::move(resp));
        return;
    }

    auto device = static_cast<tensorflow::Device*>(m_context->ctx()->device());
    if (!device) {
        ERR("Got nullptr for device in context");
        auto resp = std::make_unique<executor::RunResponse>();
        resp->mutable_result()->set_code(-1);
        cb(std::move(resp));
        return;
    }

    // NOTE: *this might be deleted by the time done_cb got called. So move out the pieces we need.
    // NOTE: both done and pContext are CopyConstructable, thus the done_cb is CopyConstructable,
    // because move-only lambda can't be assigned to std::function
    auto done_cb = [done = std::move(cb), pContext = m_context, pSession = m_exec->session()]() mutable {
        INFO("OpKernel->ComputeAsync finished with status {}", pContext->ctx()->status());

        auto resp = std::make_unique<executor::RunResponse>();
        auto tfupd = pSession->finalizeContext(pContext.get());
        tfupd.SerializeToString(resp->mutable_context()->mutable_extra());
        done(std::move(resp));
    };

    try {
        device->ComputeAsync(m_opkernel->AsAsync(), m_context->ctx(), std::move(done_cb));
//         m_opkernel->AsAsync()->ComputeAsync(pContext->ctx(), std::move(done_cb));
    } catch (std::exception &err) {
        ERR("Caught exception when run kernel compute async: ", err.what());
    }

    DEBUG("ComputeAsync returned for opkernel ", m_opkernel->name());

    // Send out a message for any pending rendezvous recv request immediately
    auto pending = m_context->rendez->releasePendingRecv();
    if (pending.size() != 0) {
        auto reqs = std::make_unique<executor::TFRendezRecvRequests>();
        for (auto &elem : pending) {
            INFO("Found pending recv request: ", elem.first);
            reqs->add_key(elem.first);
            reqs->add_allocattributes(elem.second.args.alloc_attrs.value);

            if (elem.second.args.device_context) {
                elem.second.args.device_context->Unref();
            }
        }
        m_sender->sendMessage(std::move(reqs));
    }
}

TFRunTask::~TFRunTask() = default;

PTask TFOpLibrary::createRunGraphTask(ZmqServer::Sender sender,
                                      const executor::EvenlopDef &evenlop,
                                      const executor::RunGraphRequest &request)
{
    UNUSED(sender);

    auto sess = findSession(evenlop.sessionid());
    if (!sess) { return {}; }

    tensorflow::GraphDef graphdef;
    if (!graphdef.ParseFromString(request.computation().extra())) {
        ERR("Malformated graphdef from RunGraphRequest");
    }

    return make_lambda_task([sess, graphdef = std::move(graphdef)]() mutable {
        auto resp = std::make_unique<executor::RunGraphResponse>();

        auto execState = sess->prepareExecution(std::move(graphdef));
        if (!execState) {
            ERR("Failed to get execution state");
            resp->mutable_result()->set_code(-1);
            return resp;
        }

        resp->set_execid(execState->execId());
        return resp;
    });
}

PTask TFOpLibrary::createCustomTask(ZmqServer::Sender sender, const rpc::EvenlopDef &evenlop,
                                    const rpc::CustomRequest &req)
{
    // NOTE: this-> is need to workaround a bug in GCC 6.x where member function lookup is broken
    // for generic lambda. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61636
    using Method = std::function<PTask(ZmqServer::Sender, const rpc::EvenlopDef &evenlop,
                                       const rpc::CustomRequest &)>;
    static std::unordered_map<std::string, Method> funcs{
        {"executor.TFRendezRecvUpdate", [this](auto sender, const auto &evenlop, const auto &req) {
            return this->createRendezRecvTask(std::move(sender), evenlop, req);
        }},
        {"executor.TFSessionArgs", [this](auto sender, const auto &evenlop, const auto &req) {
            return this->createInitSessionTask(std::move(sender), evenlop, req);
        }},
        {"executor.TFSessionClose", [this](auto sender, const auto &evenlop, const auto &req) {
            return this->createCloseSessionTask(std::move(sender), evenlop, req);
        }},
    };

    auto it = funcs.find(req.type());
    if (it == funcs.end()) {
        ERR("{} not found in registered custom tasks", req.type());
        return nullptr;
    }

    DEBUG("Dispatching custom task {} of seq {}", it->first, evenlop.seq());
    return it->second(std::move(sender), evenlop, req);
}

PTask TFOpLibrary::createRendezRecvTask(ZmqServer::Sender sender, const rpc::EvenlopDef &evenlop,
                                        const rpc::CustomRequest &req)
{
    UNUSED(sender);

    auto recvupd = utils::createMessage<rpc::TFRendezRecvUpdate>("executor.TFRendezRecvUpdate",
                                                                 req.extra().data(),
                                                                 req.extra().size());
    if (!recvupd) {
        return {};
    }

    auto sess = findSession(evenlop.sessionid());
    if (!sess) {
        ERR("RendezRecv request received before any run request");
        return {};
    }

    return make_lambda_task([sess, recvupd = std::move(recvupd)]() -> ProtoPtr {
        DEBUG("executor.TFRendezRecvUpdate for seq {}", recvupd->forseq());
        auto tfctx = sess->findContext(recvupd->forseq());
        if (!tfctx) {
            ERR("Context for given seq {} not found", recvupd->forseq());
            return {};
        }
        for (int i = 0; i != recvupd->items_size(); ++i) {
            auto item = recvupd->items(i);

            tensorflow::Rendezvous::ParsedKey parsed;
            auto ok = tensorflow::Rendezvous::ParseKey(item.key(), &parsed);
            if (!ok.ok()) {
                ERR("Failed to parsekey: {}", ok.error_message());
                return {};
            }

            tensorflow::Tensor tensor;
            tensorflow::AllocatorAttributes attr;

            // TODO: find a way to directly create tensor on device.
            // Currently force to create on CPU, because otherwise the tensor
            // may end up on different device than the one encoded in key, which
            // may cause problem.
            attr.set_on_host(true);
            ok.Update(tfctx->params.device->MakeTensorFromProto(item.val(), attr, &tensor));
            if (!ok.ok()) {
                ERR("Failed to create tensor for rendezrecv");
                return {};
            }

            ok.Update(tfctx->rendez->triggerSend(parsed,
                                                 tensorflow::Rendezvous::Args{nullptr, attr},
                                                 tensor, item.isdead()));
        }
        return {};
    });
}

PTask TFOpLibrary::createInitSessionTask(ZmqServer::Sender sender, const rpc::EvenlopDef &evenlop,
                                         const rpc::CustomRequest &req)
{
    UNUSED(sender);
    UNUSED(evenlop);

    auto tfreq = utils::createMessage<rpc::TFSessionArgs>("executor.TFSessionArgs",
                                                          req.extra().data(),
                                                          req.extra().size());
    if (!tfreq) {
        return {};
    }

    auto sessSeq = m_sessionSeq.fetch_add(1);
    return make_lambda_task([tfreq = std::move(tfreq), sessSeq, this]() {
        auto resp = std::make_unique<executor::CustomResponse>();

        std::string session_id = "session";
        session_id += std::to_string(sessSeq);

        auto sess = getOrCreateSession(session_id, tfreq->cfgproto());
        if (!sess) {
            ERR("Session creation failed");
            resp->mutable_result()->set_code(-1);
            return resp;
        }
        INFO("Created session {}", session_id);

        executor::TFSessionCreated tfsesscreated;
        tfsesscreated.set_sessionid(session_id);
        tfsesscreated.SerializeToString(resp->mutable_extra());
        return resp;
    });
}

PTask TFOpLibrary::createCloseSessionTask(ZmqServer::Sender sender,
                                          const executor::EvenlopDef &evenlop,
                                          const executor::CustomRequest &req)
{
    UNUSED(sender);
    UNUSED(evenlop);

    auto tfclose = utils::createMessage<rpc::TFSessionClose>("executor.TFSessionClose",
                                                             req.extra().data(),
                                                             req.extra().size());
    if (!tfclose) {
        return {};
    }

    return make_lambda_task([tfclose = std::move(tfclose), this]() {
        destorySession(tfclose->sessionid());

        auto resp = std::make_unique<executor::CustomResponse>();
        return resp;
    });
}
