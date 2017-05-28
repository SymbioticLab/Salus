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

#include "tfmocks/tfdevice.h"

#include "utils/protoutils.h"
#include "utils/pointerutils.h"
#include "platform/logging.h"

#include "executor.pb.h"
#include "tfoplibrary.pb.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_segment.h>
#include <tensorflow/core/common_runtime/function.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/function.pb.h>
#include <tensorflow/core/protobuf/config.pb.h>

namespace rpc = executor;
using ::tensorflow::NodeDef;
using ::tensorflow::ConfigProto;
using ::tensorflow::FunctionDefLibrary;
using ::google::protobuf::Message;
using std::unique_ptr;

bool TFOpLibrary::accepts(const rpc::OpKernelDef& operation)
{
    return operation.oplibrary() == rpc::OpKernelDef::TENSORFLOW;
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
    }

    return sess.get();
}

unique_ptr<tensorflow::OpKernel> TFOpLibrary::kernelFromDef(const executor::OpKernelDef &opdef)
{
    auto def = utils::createMessage<executor::TFOpKernelDef>("executor.TFOpKernelDef",
                                                             opdef.extra().data(),
                                                             opdef.extra().size());

    if (!def) { return {}; }

    DEBUG("Got NodeDef {}", def->nodedef().DebugString());
    DEBUG("Got ConfigProto {}", def->cfgproto().DebugString());
    DEBUG("Got funcdeflib {}", def->funcdef().DebugString());

    // TODO: compute session id
    std::string session_id = "session_id";

    auto sess = getOrCreateSession(session_id, def->graph_def_version(), def->cfgproto(), def->funcdef());
    if (!sess) { return {}; }

    return sess->createKernel(def->nodedef());
}

unique_ptr<tensorflow::OpKernelContext> TFOpLibrary::contextFromDef(const executor::OpContextDef &ctxdef)
{
    auto def = utils::createMessage<executor::TFOpContextDef>("executor.TFOpContextDef",
                                                              ctxdef.extra().data(),
                                                              ctxdef.extra().size());
    // FIXME: create kernel context from def
    return {};
}

executor::OpContextDef TFOpLibrary::contextToDef(const tensorflow::OpKernelContext *context)
{
    // FIXME: create def from kernel context
    return {};
}

std::unique_ptr<ITask> TFOpLibrary::createTask(const rpc::OpKernelDef& opdef, const rpc::OpContextDef& ctxdef)
{
    return std::make_unique<TFTask>(this, kernelFromDef(opdef), contextFromDef(ctxdef));
}

TFSession::TFSession(TFOpLibrary *opLibrary, const tensorflow::FunctionDefLibrary &fDefLib,
                     int graphDefVersion, const tensorflow::OptimizerOptions &optimizerOpts)
    : m_oplibrary(opLibrary)
    , m_flibDef(tensorflow::OpRegistry::Global(), fDefLib)
    , m_fruntime()
    , m_device(new TFDevice)
{
    m_fruntime.reset(tensorflow::NewFunctionLibraryRuntime(
        nullptr /* DeviceMgr */, nullptr /* Env */,
        m_device.get(), graphDefVersion, &m_flibDef, optimizerOpts));
}

TFSession::~TFSession() = default;

std::unique_ptr<tensorflow::OpKernel> TFSession::createKernel(const tensorflow::NodeDef &ndef)
{
    tensorflow::OpKernel *kernel = nullptr;
    // Caches the kernel only if the node is stateful.
    if (!m_fruntime->IsStateful(ndef.op())) {
        auto ok = m_fruntime->CreateKernel(ndef, &kernel);
        if (!ok.ok()) {
            ERR("Failed to create kernel with status {}({}) for NodeDef: {}",
                ok.code(), ok.error_message(), ndef.DebugString());
        }
        return std::unique_ptr<tensorflow::OpKernel>(kernel);
    }

    // Kernels created for subgraph nodes need to be cached.  On
    // cache miss, create_fn() is invoked to create a kernel based
    // on the function library here + global op registry.
    auto lib = m_fruntime.get();
    auto create_fn = [lib, &ndef](tensorflow::OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
    };
    auto ok = m_opseg.FindOrCreate("executor_session", ndef.name(), &kernel, create_fn);

    return std::unique_ptr<tensorflow::OpKernel>(kernel);
}

TFTask::TFTask(TFOpLibrary *library, unique_ptr<tensorflow::OpKernel> &&kernel,
               unique_ptr<tensorflow::OpKernelContext> &&context)
    : m_opkernel(std::move(kernel))
    , m_context(std::move(context))
    , m_library(library)
{
    INFO("Created TFTask.");
    if (m_opkernel) {
        INFO("m_opkernel.def() {}", m_opkernel->def().DebugString());
        INFO("m_opkernel.name() {}", m_opkernel->name());
        INFO("m_opkernel.type_string() {}", m_opkernel->type_string());
        INFO("m_opkernel.is_internal() {}", m_opkernel->is_internal());
        INFO("m_opkernel.num_inputs() {}", m_opkernel->num_inputs());
        for (int i = 0; i != m_opkernel->num_inputs(); i++) {
            INFO("m_opkernel.input_type({}) {}", i, m_opkernel->input_type(i));
        }
        for (int i = 0; i != m_opkernel->input_memory_types().size(); i++) {
            INFO("m_opkernel.input_memory_types()[{}] {}", i, m_opkernel->input_memory_types()[i]);
        }

        INFO("m_opkernel.num_outputs() {}", m_opkernel->num_outputs());
        for (int i = 0; i != m_opkernel->num_outputs(); i++) {
            INFO("m_opkernel.output_type({}) {}", i, m_opkernel->output_type(i));
        }
        for (int i = 0; i != m_opkernel->output_memory_types().size(); i++) {
            INFO("m_opkernel.output_memory_types()[{}] {}", i, m_opkernel->output_memory_types()[i]);
        }

        INFO("m_opkernel.IsExpensive() {}", m_opkernel->IsExpensive());
    }

    if (m_context) {
        INFO("");
    }
}

rpc::OpContextDef TFTask::contextDef()
{
    return m_library->contextToDef(m_context.get());
}

rpc::Status TFTask::run()
{
    if (m_opkernel && m_context) {
        m_opkernel->Compute(m_context.get());
    } else {
        ERR("Got nullptr for opkernel or context: m_opkernel = {:x}, m_context = {:x}",
            reinterpret_cast<uint64_t>(m_opkernel.get()), reinterpret_cast<uint64_t>(m_context.get()));
    }

    // TODO: proper return code
    return {};
}

