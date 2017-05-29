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
#include <tensorflow/core/framework/op_segment.h>
#include <tensorflow/core/common_runtime/function.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/function.pb.h>
#include <tensorflow/core/protobuf/config.pb.h>
#include <tensorflow/core/lib/gtl/stl_util.h>

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

TFSession *TFOpLibrary::getSession(const std::string& sess_id)
{
    std::lock_guard<std::mutex> guard(m_mu);
    if (m_sessions.count(sess_id) != 0) {
        return m_sessions[sess_id].get();
    }
    return {};
}

std::unique_ptr<ITask> TFOpLibrary::createTask(const rpc::OpKernelDef& opdef, const rpc::OpContextDef& ctxdef)
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
    TRACE("Created OpKernel");
    auto tfctx = sess->createContext(*tfctxdef, opkernel.get());
    TRACE("Created OpKernelContext");

    return std::make_unique<TFTask>(this, std::move(opkernel), std::move(tfctx));
}

TFSession::TFSession(TFOpLibrary *opLibrary, const tensorflow::FunctionDefLibrary &fDefLib,
                     int graphDefVersion, const tensorflow::OptimizerOptions &optimizerOpts)
    : m_oplibrary(opLibrary)
    , m_flibDef(tensorflow::OpRegistry::Global(), fDefLib)
    , m_fruntime(nullptr)
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

TFContext::TFContext()
    : step_container(0, [](const std::string&) {})
{ }

TFContext::~TFContext() { }

tensorflow::OpKernelContext *TFContext::ctx()
{
    if (!context) {
        context.reset(new tensorflow::OpKernelContext(&params));
    }
    return context.get();
}

inline void TFContext::FillOutputAttrs() {
    output_attrs.clear();
    for (int index = 0; index < params.op_kernel->num_outputs(); index++) {
        tensorflow::AllocatorAttributes attr;
        const bool on_host =
        (params.op_kernel->output_memory_types()[index] == tensorflow::HOST_MEMORY);
        attr.set_on_host(on_host);
        output_attrs.push_back(attr);
    }
    params.output_attr_array = tensorflow::gtl::vector_as_array(&output_attrs);
}

std::unique_ptr<TFContext> TFSession::createContext(const executor::TFOpContextDef &tfdef,
                                                    tensorflow::OpKernel *opkernel)
{
    auto tfctx = std::make_unique<TFContext>();

    tfctx->params.device = m_device.get();
    tfctx->params.op_kernel = opkernel;
    tfctx->params.step_container = &tfctx->step_container;
    tfctx->params.slice_reader_cache = &tfctx->slice_reader_cache_wrapper;
    tfctx->params.resource_manager = m_device->resource_manager();
    tfctx->params.function_library = m_fruntime.get();

    tfctx->params.step_id = tfdef.step_id();
    tfctx->params.frame_iter = tensorflow::FrameAndIter(tfdef.frame_id(), tfdef.iter_id());
    tfctx->params.is_input_dead = tfdef.is_input_dead();
    tfctx->FillOutputAttrs();

    tfctx->params.inputs = &tfctx->inputs;
    tfctx->params.input_device_contexts = &tfctx->input_device_contexts;
    tfctx->params.input_alloc_attrs = &tfctx->input_alloc_attrs;
    // FIXME: prepare inputs

    return tfctx;
}

TFTask::TFTask(TFOpLibrary *library, unique_ptr<tensorflow::OpKernel> &&kernel,
               unique_ptr<TFContext> &&context)
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
    return m_library->contextToDef(m_context->ctx());
}

rpc::Status TFTask::run()
{
    if (m_opkernel && m_context) {
        m_opkernel->Compute(m_context->ctx());
    } else {
        ERR("Got nullptr for opkernel or context: m_opkernel = {:x}, m_context = {:x}",
            reinterpret_cast<uint64_t>(m_opkernel.get()), reinterpret_cast<uint64_t>(m_context.get()));
    }

    INFO("OpKernel->Compute finished");

    // TODO: proper return code
    return {};
}

executor::OpContextDef TFOpLibrary::contextToDef(tensorflow::OpKernelContext *context)
{
    executor::TFOpContextDef tfctxdef;
    if (!context->status().ok()) {
        tfctxdef.set_status_code(context->status().code());
        tfctxdef.set_status_msg(context->status().error_message());
    }
    tfctxdef.set_is_output_dead(*context->is_output_dead());

    // FIXME: set outputs


    executor::OpContextDef def;
    tfctxdef.SerializeToString(def.mutable_extra());
    return def;
}
