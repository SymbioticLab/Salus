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

#include "tfoplibrary.h"

#include "tfmocks/tfsession.h"

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

#include <thread>
#include <functional>

namespace rpc = executor;
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
}

void dumpOpContext(tensorflow::OpKernelContext *ctx)
{
    if (!ctx) return;

    TRACE("context.is_output_dead() {}", *ctx->is_output_dead());
    TRACE("context.num_outputs() {}", ctx->num_outputs());
}

} // namespace

REGISTER_OPLIBRARY(executor::TENSORFLOW, TFOpLibrary);

TFOpLibrary::~TFOpLibrary() = default;

bool TFOpLibrary::accepts(const rpc::OpKernelDef& operation)
{
    return operation.oplibrary() == rpc::TENSORFLOW;
}

TFSession *TFOpLibrary::getOrCreateSession(const std::string& sess_id, int graph_def_version,
                                           const tensorflow::ConfigProto& cfgProto,
                                           const tensorflow::FunctionDefLibrary& fDefLib)
{
    std::lock_guard<std::mutex> guard(m_mu);

    auto &sess = m_sessions[sess_id];
    if (!sess) {
        sess = std::make_unique<TFSession>(this, fDefLib, graph_def_version, cfgProto);
    } else {
        DEBUG("Reuse previously created session");
    }

    return sess.get();
}

TFSession *TFOpLibrary::getSession(const std::string& sess_id)
{
    std::lock_guard<std::mutex> guard(m_mu);
    if (m_sessions.count(sess_id) != 0) {
        return m_sessions[sess_id].get();
    }
    return {};
}

PTask TFOpLibrary::createRunTask(ZmqServer::Sender sender, const rpc::OpKernelDef& opdef, const rpc::OpContextDef& ctxdef)
{
    auto tfdef = utils::createMessage<executor::TFOpKernelDef>("executor.TFOpKernelDef",
                                                             opdef.extra().data(),
                                                             opdef.extra().size());
    auto tfctxdef = utils::createMessage<executor::TFOpContextDef>("executor.TFOpContextDef",
                                                                   ctxdef.extra().data(),
                                                                   ctxdef.extra().size());

    if (!tfdef || !tfctxdef) { return {}; }

    DEBUG("Got isAsync {}", tfdef->isasync());
    DEBUG("Got NodeDef {}", tfdef->nodedef().DebugString());
    DEBUG("Got ConfigProto {}", tfdef->cfgproto().DebugString());
    DEBUG("Got funcdeflib {}", tfdef->funcdef().DebugString());

    // TODO: compute session id
    std::string session_id = "session_id";

    auto sess = getOrCreateSession(session_id, tfdef->graph_def_version(), tfdef->cfgproto(), tfdef->funcdef());
    if (!sess) { return {}; }

    auto ndef = std::unique_ptr<NodeDef>(tfdef->release_nodedef());
    return std::make_unique<TFRunTask>(sess, std::move(sender), std::move(ndef),
                                       tfdef->isasync(), std::move(tfctxdef));
}

TFRunTask::TFRunTask(TFSession *sess, ZmqServer::Sender &&sender, unique_ptr<NodeDef> &&nodedef, bool async,
                     unique_ptr<executor::TFOpContextDef> &&tfctxdef)
    : m_session(sess)
    , m_sender(std::move(sender))
    , m_ndef(std::move(nodedef))
    , m_tfctxdef(std::move(tfctxdef))
    , m_async(async)
{
}

bool TFRunTask::isAsync()
{
    return m_async;
}

bool TFRunTask::prepare(DeviceType dev)
{
    UNUSED(dev);

    auto kernel = m_session->findOrCreateKernel(*m_ndef);
    if (m_async) {
        m_opkernel = kernel->AsAsync();
    } else {
        m_opkernel = kernel;
    }
    if (!m_opkernel) {
        ERR("Kernel creation failed, got {:x}, after conversion {:x}",
            reinterpret_cast<uint64_t>(kernel), reinterpret_cast<uint64_t>(m_opkernel));
        return false;
    }
    INFO("Created OpKernel");
    dumpOpKernel(m_opkernel);

    m_context = m_session->createContext(*m_tfctxdef, m_opkernel);
    if (!m_context) {
        return false;
    }
    TRACE("Created OpKernelContext");
    dumpOpContext(m_context->ctx());

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

    try {
        m_opkernel->Compute(m_context->ctx());
    } catch (std::exception &err) {
        ERR("Caught exception when run kernel compute: ", err.what());
    }

    INFO("OpKernel->Compute finished with status {}", m_context->ctx()->status());

    auto ctxdef = resp->mutable_context();

    auto context = m_context->ctx();
    rpc::TFOpContextUpdate tfctxupd;
    tfctxupd.set_status_code(context->status().code());
    tfctxupd.set_status_msg(context->status().error_message());
    tfctxupd.set_is_output_dead(*context->is_output_dead());

    // process tensor received by rendezvous
    // Note that rendezvous already registered these tensors to m_session
    // And this will clear tensors table in rendezvous
    for (auto &elem : m_session->rendezvous().releasePendingSentTensors()) {
        auto item = tfctxupd.add_rendeztensors();
        item->set_key(elem.first);
        item->set_allocattributes(elem.second.args.alloc_attrs.value);
        item->set_isdead(elem.second.isDead);
        m_session->tensorMetaToProto(item->mutable_val(), elem.second.val);
    }

    // process tensor set as outputs
    for (int i = 0; i != context->num_outputs(); i++) {
        auto out = context->release_output(i);
        auto outdef = tfctxupd.add_outputs();

        // Let the session manage the tensor memory
        m_session->registerTensorMemory(*out.tensor);

        // TODO: handle ref TensorValue
        m_session->tensorMetaToProto(outdef, *out.tensor);
        if (!out.is_ref()) {
            delete out.tensor;
        }
    }

    // write update to ctxdef
    tfctxupd.SerializeToString(ctxdef->mutable_extra());

    return resp;
}

void TFRunTask::runAsync(DoneCallback cb)
{
    // FIXME: implement this
    INFO("running async in thread {}", std::this_thread::get_id());

    if (!m_opkernel || !m_context) {
        ERR("Got nullptr for opkernel or context: m_opkernel = {:x}, m_context = {:x}",
            reinterpret_cast<uint64_t>(m_opkernel), reinterpret_cast<uint64_t>(m_context.get()));
        auto resp = std::make_unique<executor::RunResponse>();
        resp->mutable_result()->set_code(-1);
        cb(std::move(resp));
        return;
    }

    auto ctx = m_context->ctx(); // we need this later
    auto pSession = m_session;
    // NOTE: this might be deleted by the time done_cb got called. So move out the pieces we need.
    // NOTE: both done and pContext are CopyConstructable, thus the done_cb is CopyConstructable,
    // because move-only lambda can't be assigned to std::function
    auto done_cb = [done = std::move(cb), pContext = std::move(m_context), pSession]() mutable {
        INFO("OpKernel->ComputeAsync finished with status {}", pContext->ctx()->status());

        auto resp = std::make_unique<executor::RunResponse>();
        auto ctxdef = resp->mutable_context();

        auto context = pContext->ctx();
        rpc::TFOpContextUpdate tfctxupd;
        tfctxupd.set_status_code(context->status().code());
        tfctxupd.set_status_msg(context->status().error_message());
        tfctxupd.set_is_output_dead(*context->is_output_dead());

        // process tensor received by rendezvous
        // Note that rendezvous already registered these tensors to m_session
        // And this will clear tensors table in rendezvous
        for (auto &elem : pSession->rendezvous().releasePendingSentTensors()) {
            auto item = tfctxupd.add_rendeztensors();
            item->set_key(elem.first);
            item->set_allocattributes(elem.second.args.alloc_attrs.value);
            item->set_isdead(elem.second.isDead);
            pSession->tensorMetaToProto(item->mutable_val(), elem.second.val);
        }

        // process tensor set as outputs
        for (int i = 0; i != context->num_outputs(); i++) {
            auto out = context->release_output(i);
            auto outdef = tfctxupd.add_outputs();

            // Let the session manage the tensor memory
            pSession->registerTensorMemory(*out.tensor);

            // TODO: handle ref TensorValue
            if (out.is_ref()) {
                WARN("Ref tensor handling missing!!!!!");
            }
            pSession->tensorMetaToProto(outdef, *out.tensor);
            if (!out.is_ref()) {
                delete out.tensor;
            }
        }

        // write update to ctxdef
        tfctxupd.SerializeToString(ctxdef->mutable_extra());

        done(std::move(resp));
    };

    try {
        m_opkernel->AsAsync()->ComputeAsync(ctx, std::move(done_cb));
    } catch (std::exception &err) {
        ERR("Caught exception when run kernel compute async: ", err.what());
    }

    // Send out a message for any pending rendezvous recv request immediately
    auto reqs = std::make_unique<executor::TFRendezRecvRequests>();
    for (auto &elem : m_session->rendezvous().releasePendingRecv()) {
        INFO("Found pending recv request: ", elem.first);
        reqs->add_key(elem.first);
        reqs->add_allocattributes(elem.second.args.alloc_attrs.value);
    }
    m_sender->sendMessage(std::move(reqs));
}

TFRunTask::~TFRunTask() = default;

PTask TFOpLibrary::createFetchTask(ZmqServer::Sender sender, const executor::CustomRequest &fetch)
{
    UNUSED(sender);

    auto tftensors = utils::createMessage<executor::TFTensors>("executor.TFTensors",
                                                               fetch.extra().data(),
                                                               fetch.extra().size());
    if (!tftensors) {
        return {};
    }

    // TODO: compute session id
    std::string session_id = "session_id";
    auto sess = getSession(session_id);
    if (!sess) {
        ERR("Fetch request received before any run request");
        return {};
    }
    return std::make_unique<TFFetchTask>(sess, std::move(tftensors));
}

TFFetchTask::TFFetchTask(TFSession* session, std::unique_ptr<executor::TFTensors> && tensors)
    : m_tensorMetas(std::move(tensors))
    , m_session(session)
{
}

ProtoPtr TFFetchTask::run()
{
    auto resp = std::make_unique<executor::CustomResponse>();

    executor::TFTensors full_tensors;
    for (auto &proto : m_tensorMetas->tensors()) {
        auto tensor = m_session->findTensorFromProtoMeta(proto);
        if (!tensor) {
            ERR("Requested tensor not found in this session: {}", proto.DebugString());
            resp->mutable_result()->set_code(-1);
            return resp;
        }
        INFO("Found a tensor: {}", tensor->DebugString());
        tensor->AsProtoTensorContent(full_tensors.add_tensors());
    }
    full_tensors.SerializeToString(resp->mutable_extra());

    return resp;
}

TFFetchTask::~TFFetchTask() = default;

PTask TFOpLibrary::createPushTask(ZmqServer::Sender sender, const executor::CustomRequest &push)
{
    UNUSED(sender);

    auto tfpush = utils::createMessage<executor::TFPushRequest>("executor.TFPushRequest",
                                                                push.extra().data(),
                                                                push.extra().size());

    if (!tfpush) {
        return {};
    }

    // TODO: compute session id
    std::string session_id = "session_id";
    auto sess = getSession(session_id);
    if (!sess) {
        ERR("Push request received before any run request");
        return {};
    }
    return std::make_unique<TFPushTask>(sess, std::move(tfpush));
}

TFPushTask::TFPushTask(TFSession* session, std::unique_ptr<executor::TFPushRequest> && tensors)
    : m_tensors(std::move(tensors))
    , m_session(session)
{
}

ProtoPtr TFPushTask::run()
{
    auto resp = std::make_unique<executor::CustomResponse>();
    if (m_tensors->data_size() != m_tensors->tensors_size()) {
        ERR("Number of tensors mismatch in push request: data {}, tensors {}",
            m_tensors->data_size(), m_tensors->tensors_size());
        resp->mutable_result()->set_code(-1);
        return resp;
    }

    for (int i = 0; i != m_tensors->tensors_size(); ++i) {
        auto t = m_session->findTensorFromProtoMeta(m_tensors->tensors(i));

        auto &dataproto = m_tensors->data(i);
        if (!m_session->isCompatible(*t, dataproto)) {
            ERR("Tensor not compatible with pushed data tensor proto");
            continue;
        }
        if (!t->FromProto(dataproto)) {
            ERR("Malformated tensor proto");
            continue;
        }
    }

    return resp;
}

TFPushTask::~TFPushTask() = default;

PTask TFOpLibrary::createCustomTask(ZmqServer::Sender sender, const executor::CustomRequest &req)
{
    // NOTE: this-> is need to workaround a bug in GCC 6.x where member function lookup is broken
    // for generic lambda. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61636
    using Method = std::function<PTask(ZmqServer::Sender, const executor::CustomRequest &)>;
    static std::unordered_map<std::string, Method> funcs{
        {"executor.TFPushTask", [this](auto sender, const auto &req) {
            return this->createPushTask(std::move(sender), req);
        }},
        {"executor.TFFetchTask", [this](auto sender, const auto &req) {
            return this->createFetchTask(std::move(sender), req);
        }},
        {"executor.TFRendezRecvResponse", [this](auto sender, const auto &req) {
            return this->createRendezRecvTask(std::move(sender), req);
        }},
    };

    assert(funcs.count(req.type()) == 1);

    return funcs[req.type()](std::move(sender), req);
}

class TFRendezRecvTask : public ITask
{
public:
    ~TFRendezRecvTask() override = default;

    TFRendezRecvTask(TFSession *session, std::unique_ptr<executor::TFRendezRecvResponse> &&recv)
        : m_recv(std::move(recv))
        , m_session(session)
    { }

    ProtoPtr run() override
    {
        for (int i = 0; i != m_recv->items_size(); ++i) {
            auto item = m_recv->items(i);

            tensorflow::Rendezvous::ParsedKey parsed;
            auto ok = tensorflow::Rendezvous::ParseKey(item.key(), &parsed);
            ok.Update(m_session->rendezvous().Send(parsed,
                                                   tensorflow::Rendezvous::Args(),
                                                   *m_session->createAndRegister(item.val()),
                                                   item.isdead()));
        }
        return {};
    }

private:
    std::unique_ptr<executor::TFRendezRecvResponse> m_recv;

    TFSession *m_session;
};

PTask TFOpLibrary::createRendezRecvTask(ZmqServer::Sender sender, const executor::CustomRequest &req)
{
    UNUSED(sender);

    auto recvResp = utils::createMessage<executor::TFRendezRecvResponse>("executor.TFRendezRecvResponse",
                                                                         req.extra().data(),
                                                                         req.extra().size());

    if (!recvResp) {
        return {};
    }

    // TODO: compute session id
    std::string session_id = "session_id";
    auto sess = getSession(session_id);
    if (!sess) {
        ERR("RendezRecv request received before any run request");
        return {};
    }
    return std::make_unique<TFRendezRecvTask>(sess, std::move(recvResp));
}
