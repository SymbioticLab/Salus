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

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_segment.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/function.pb.h>
#include <tensorflow/core/protobuf/config.pb.h>

namespace rpc = executor;
using ::tensorflow::NodeDef;
using ::tensorflow::ConfigProto;
using ::tensorflow::FunctionDefLibrary;
using ::google::protobuf::Message;
using std::unique_ptr;

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

}

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
        sess.reset(new TFSession(this, fDefLib, graph_def_version,
                                 cfgProto.graph_options().optimizer_options()));
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

std::unique_ptr<ITask> TFOpLibrary::createRunTask(const rpc::OpKernelDef& opdef, const rpc::OpContextDef& ctxdef)
{
    auto tfdef = utils::createMessage<executor::TFOpKernelDef>("executor.TFOpKernelDef",
                                                             opdef.extra().data(),
                                                             opdef.extra().size());
    auto tfctxdef = utils::createMessage<executor::TFOpContextDef>("executor.TFOpContextDef",
                                                                   ctxdef.extra().data(),
                                                                   ctxdef.extra().size());

    if (!tfdef || !tfctxdef) { return {}; }

    DEBUG("Got NodeDef {}", tfdef->nodedef().DebugString());
    DEBUG("Got ConfigProto {}", tfdef->cfgproto().DebugString());
    DEBUG("Got funcdeflib {}", tfdef->funcdef().DebugString());

    // TODO: compute session id
    std::string session_id = "session_id";

    auto sess = getOrCreateSession(session_id, tfdef->graph_def_version(), tfdef->cfgproto(), tfdef->funcdef());
    if (!sess) { return {}; }

    auto opkernel = sess->createKernel(tfdef->nodedef());
    if (!opkernel) {
        return {};
    }
    INFO("Created OpKernel");
    dumpOpKernel(opkernel.get());

    auto tfctx = sess->createContext(*tfctxdef, opkernel.get());
    if (!tfctx) {
        return {};
    }
    TRACE("Created OpKernelContext");
    dumpOpContext(tfctx->ctx());

    return std::make_unique<TFRunTask>(sess, std::move(opkernel), std::move(tfctx));
}

TFRunTask::TFRunTask(TFSession *sess, unique_ptr<tensorflow::OpKernel> &&kernel,
                     unique_ptr<TFContext> &&context)
    : m_opkernel(std::move(kernel))
    , m_context(std::move(context))
    , m_session(sess)
{
}

rpc::Status TFRunTask::run(google::protobuf::Message *out)
{
    if (!m_opkernel || !m_context) {
        ERR("Got nullptr for opkernel or context: m_opkernel = {:x}, m_context = {:x}",
            reinterpret_cast<uint64_t>(m_opkernel.get()), reinterpret_cast<uint64_t>(m_context.get()));
        // TODO: proper return status
        return {};
    }

    try {
        m_opkernel->Compute(m_context->ctx());
    } catch (std::exception &err) {
        ERR("Caught exception when run kernel compute: ", err.what());
    }
    INFO("OpKernel->Compute finished with status {}", m_context->ctx()->status());

    auto ctxdef = static_cast<executor::OpContextDef*>(out);

    auto context = m_context->ctx();
    rpc::TFOpContextUpdate tfctxupd;
    tfctxupd.set_status_code(context->status().code());
    tfctxupd.set_status_msg(context->status().error_message());
    tfctxupd.set_is_output_dead(*context->is_output_dead());

    for (int i = 0; i != context->num_outputs(); i++) {
        auto out = context->release_output(i);
        auto outdef = tfctxupd.add_outputs();

        // Let the session manage the tensor memory
        m_session->registerTensorMemory(*out.tensor);

        m_session->tensorMetaToProto(outdef, *out.tensor);
        if (!out.is_ref()) {
            delete out.tensor;
        }
    }

    tfctxupd.SerializeToString(ctxdef->mutable_extra());
    // TODO: proper return code
    return {};
}

TFRunTask::~TFRunTask() = default;

std::unique_ptr<ITask> TFOpLibrary::createFetchTask(const executor::FetchRequest &fetch)
{
    auto tftensors = utils::createMessage<executor::TFTensors>("executor.TFTensors",
                                                               fetch.extra().data(),
                                                               fetch.extra().size());

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
    : m_tensors(std::move(tensors))
    , m_session(session)
{
}

executor::Status TFFetchTask::run(google::protobuf::Message* out)
{
    executor::TFTensors ret;

    for (auto &proto : m_tensors->tensors()) {
        auto tensor = m_session->findTensorFromProto(proto);
        if (!tensor) {
            // TODO: proper return status
            return {};
        }
        tensor->AsProtoTensorContent(ret.add_tensors());
    }

    auto resp = static_cast<executor::FetchResponse*>(out);
    ret.SerializeToString(resp->mutable_extra());

    // TODO: proper return status
    return {};
}

TFFetchTask::~TFFetchTask() = default;

std::unique_ptr<ITask> TFOpLibrary::createPushTask(const executor::PushRequest &push)
{
    auto tfpush = utils::createMessage<executor::TFPushRequest>("executor.TFPushRequest",
                                                                push.extra().data(),
                                                                push.extra().size());

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

executor::Status TFPushTask::run(google::protobuf::Message *out)
{
    UNUSED(out);

    if (m_tensors->data_size() != m_tensors->tensors_size()) {
        ERR("Number of tensors mismatch in push request: data {}, tensors {}",
            m_tensors->data_size(), m_tensors->tensors_size());
    }

    for (int i = 0; i != m_tensors->tensors_size(); ++i) {
        auto t = m_session->findTensorFromProto(m_tensors->tensors(i));

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

    // TODO: proper return status
    return {};
}

TFPushTask::~TFPushTask() = default;
