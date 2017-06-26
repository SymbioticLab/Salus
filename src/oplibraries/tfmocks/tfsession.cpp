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

#include "tfsession.h"
#include "tfdevice.h"
#include "tfallocator.h"
#include "tfrendezvous.h"

#include "platform/logging.h"
#include "memorymgr/memorymgr.h"
#include "utils/macros.h"

#include "tfoplibrary.pb.h"

#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/common_runtime/function.h>
#include <tensorflow/core/lib/gtl/stl_util.h>

TFSession::TFSession(TFOpLibrary *opLibrary, const tensorflow::ConfigProto &configProto)
    : m_oplibrary(opLibrary)
    , m_sessHandle("executor_session")
    , m_options()
    , m_allocator(TFAllocator::New())
    , m_opseg()
{
    DEBUG("Creating new TFSession at {:x}", reinterpret_cast<uint64_t>(this));

    m_options.config = configProto;

    m_device = std::make_unique<TFDevice>(m_options, m_allocator);

    m_opseg.AddHold(m_sessHandle);
}

TFSession::~TFSession()
{
    m_opseg.RemoveHold(m_sessHandle);
    for (auto p : m_tensors) {
        if (!p.second.is_ref()) {
            delete p.second.tensor;
        }
    }
}

std::unique_ptr<tensorflow::Tensor> TFSession::tensorFromProtoData(const tensorflow::TensorProto &proto)
{
    if (!m_device) {
        ERR("m_device should not be nullptr for TFSession");
        return nullptr;
    }

    auto tensor = std::make_unique<tensorflow::Tensor>();

    auto status = m_device->MakeTensorFromProto(proto, {}, tensor.get());
    if (!status.ok()) {
        ERR("Error when create tensor");
        tensor.reset();
    }
    return tensor;
}

bool TFSession::findTensorFromName(const std::string &name, TensorValue &val)
{
    tensorflow::mutex_lock locker(m_mu);
    auto it = m_tensors.find(name);
    if (it == m_tensors.end()) {
        ERR("Tensor not found under name: {}", name);
        return false;
    }
    val = it->second;
    return true;
}

void TFSession::registerTensorForName(const std::string &name, TensorValue val)
{
    INFO("Registering tensor: {}, is ref: {} under name: {}",
         val->DebugString(), val.is_ref(), name);
    tensorflow::mutex_lock locker(m_mu);
    auto it = m_tensors.find(name);
    if (it == m_tensors.end()) {
        m_tensors.emplace(name, val);
    } else {
        if (it->second.mutex_if_ref != val.mutex_if_ref) {
            WARN("The tensor going to be registered already exists, and is under a different mutex");
        }
        it->second = val;
    }
}

bool TFSession::isCompatible(const tensorflow::Tensor &tensor, const tensorflow::TensorProto &proto) const
{
    auto dtype = proto.dtype();
    if (tensorflow::IsRefType(dtype)) {
        dtype = tensorflow::RemoveRefType(proto.dtype());
    }
    tensorflow::TensorShape shape(proto.tensor_shape());
    if (tensor.dtype() != dtype
        || tensor.shape() != shape) {
        ERR("Requested tensor metadata mismatch with record. Requested: {} of type {}, stored: {} of type {}",
            tensor.shape().DebugString(), tensor.dtype(),
            shape.DebugString(), proto.dtype());
        return false;
    }
    return true;
}

void TFSession::tensorToProtoMeta(tensorflow::TensorProto *meta, TensorValue val)
{
    meta->set_dtype(val->dtype());

    MaybeLock locker(val);
    val->shape().AsProto(meta->mutable_tensor_shape());

    if (val->IsInitialized() && val->shape().num_elements() > 0) {
        auto addr_handle = reinterpret_cast<uint64_t>(val->tensor_data().data());
        // HACK: use a int64 val entry to store the addr handle for simplicity,
        // idealy should store this in tensor_content with proper encoding.
        meta->add_int64_val(addr_handle);
    }
}

void TFSession::tensorToProtoData(tensorflow::TensorProto *data, TensorValue val)
{
    MaybeLock locker(val);
    val->AsProtoTensorContent(data);
}

TFExecutionState *TFSession::prepareExecution(tensorflow::GraphDef &&graphdef)
{
    static std::atomic_uint_fast64_t counter(0);
    std::string execId("executor");
    execId += std::to_string(counter.fetch_add(1));

    auto e = findExecution(execId);
    if (e) {
        return e;
    }

    auto execState = std::make_unique<TFExecutionState>(this, execId, std::move(graphdef),
                                                        m_options.config.graph_options().optimizer_options());
    tensorflow::mutex_lock locker(m_muexec);
    auto p = m_execStates.emplace(execId, std::move(execState));
    return p.first->second.get();
}

TFExecutionState *TFSession::findExecution(const std::string &execId)
{
    tensorflow::mutex_lock locker(m_muexec);
    auto it = m_execStates.find(execId);
    if (it != m_execStates.end()) {
        return it->second.get();
    }
    ERR("Execution {} not found in session {}", execId, m_sessHandle);
    return nullptr;
}

TFExecutionState::TFExecutionState(TFSession *sess, const std::string &execId, tensorflow::GraphDef &&graph,
                                   const tensorflow::OptimizerOptions &optOptions)
    : m_session(sess)
    , m_execId(execId)
    , m_graphdef(std::move(graph))
    , m_rendez(tensorflow::NewLocalRendezvous())
    , m_fdefinition(nullptr)
    , m_fruntime(nullptr)
{
    m_fdefinition = std::make_unique<tensorflow::FunctionLibraryDefinition>(tensorflow::OpRegistry::Global(),
                                                                            m_graphdef.library());
    m_fruntime.reset(tensorflow::NewFunctionLibraryRuntime(nullptr,
                                                           m_session->m_options.env,
                                                           m_session->m_device.get(),
                                                           m_graphdef.versions().producer(),
                                                           m_fdefinition.get(), optOptions));
}

const std::string &TFExecutionState::execId() const
{
    return m_execId;
}

TFSession *TFExecutionState::session()
{
    return m_session;
}

tensorflow::FunctionLibraryRuntime *TFExecutionState::functionRuntime()
{
    return m_fruntime.get();
}

tensorflow::Rendezvous *TFExecutionState::rendez()
{
    return m_rendez;
}

TFExecutionState::~TFExecutionState()
{
    m_rendez->Unref();
}

tensorflow::OpKernel *TFSession::findOrCreateKernel(const tensorflow::NodeDef &ndef,
                                                    TFExecutionState *execState)
{
    tensorflow::OpKernel *kernel = nullptr;
    // Caches the kernel only if the node is stateful.
    auto fruntime = execState->functionRuntime();
    if (!fruntime->IsStateful(ndef.op())) {
        auto ok = fruntime->CreateKernel(ndef, &kernel);
        if (!ok.ok()) {
            ERR("Failed to create kernel with status {} for NodeDef: {}", ok,
                ndef.DebugString());
            delete kernel;
            kernel = nullptr;
        }
        if (kernel) {
            m_kernels.emplace_back(kernel);
        }
        return kernel;
    }

    // Kernels created for subgraph nodes need to be cached.  On
    // cache miss, create_fn() is invoked to create a kernel based
    // on the function library here + global op registry.
    // OpSegment takes ownership of the created kernel.
    auto create_fn = [fruntime, &ndef](tensorflow::OpKernel** kernel) {
        return fruntime->CreateKernel(ndef, kernel);
    };
    auto ok = m_opseg.FindOrCreate(m_sessHandle, ndef.name(), &kernel, create_fn);
    if (!ok.ok()) {
        ERR("Failed to create kernel with status {} for NodeDef: {}", ok,
            ndef.DebugString());
    }

    return kernel;
}

std::unique_ptr<TFContext> TFSession::createContext(const executor::TFOpContextDef &tfdef,
                                                    tensorflow::OpKernel *opkernel, uint64_t seq,
                                                    TFExecutionState *execState)
{
    auto tfctx = std::make_unique<TFContext>(execState, seq);
    registerContext(seq, tfctx.get());

    tfctx->params.device = m_device.get();
    tfctx->params.op_kernel = opkernel;
    tfctx->params.step_container = &tfctx->step_container;
    tfctx->params.slice_reader_cache = &tfctx->slice_reader_cache_wrapper;
    tfctx->params.resource_manager = m_device->resource_manager();
    tfctx->params.function_library = execState->functionRuntime();
    tfctx->params.rendezvous = &tfctx->rendez;
    tfctx->params.tensor_store = &tfctx->tensorStore;

    tfctx->params.step_id = tfdef.step_id();
    tfctx->params.frame_iter = tensorflow::FrameAndIter(tfdef.frame_id(), tfdef.iter_id());
    tfctx->params.is_input_dead = tfdef.is_input_dead();
    tfctx->FillOutputAttrs();

    tfctx->FillInputAttrs();
    tfctx->FillInputDeviceContext();

    auto num_inputs = opkernel->num_inputs();
    if (num_inputs != tfdef.inputs_size()) {
        ERR("Missing inputs in received TFOpContextDef: required {}, found {}",
            num_inputs, tfdef.inputs_size());
        return {};
    }
    tfctx->inputs.reserve(num_inputs);
    for (int i = 0; i != tfdef.inputs_size(); ++i) {
        const auto &initem = tfdef.inputs(i);
        if (initem.name() != opkernel->def().input(i)) {
            ERR("Mismatch input: {}, expected: {}", initem.name(), opkernel->def().input(i));
            return {};
        }
        TensorValue input;
        if (!findTensorFromName(initem.name(), input)) {
            ERR("Input not found");
            return {};
        }
        if (initem.is_ref() && !input.is_ref()) {
            ERR("{}-th input expects a ref type", i);
            return {};
        }
        tfctx->inputs.push_back(input);
    }
    tfctx->params.inputs = &tfctx->inputs;

    return tfctx;
}

TFContext::TFContext(TFExecutionState *exec, uint64_t seq)
    : seq(seq)
    , step_container(0, [](const std::string&) {})
    , rendez(exec)
    , m_exec(exec)
{
}

TFContext::~TFContext() = default;

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

inline void TFContext::FillInputAttrs()
{
    input_alloc_attrs.clear();
    input_alloc_attrs.reserve(params.op_kernel->num_inputs());
    for (int index = 0; index < params.op_kernel->num_inputs(); index++) {
        tensorflow::AllocatorAttributes attr;
        const bool on_host =
        (params.op_kernel->input_memory_types()[index] == tensorflow::HOST_MEMORY);
        attr.set_on_host(on_host);
        input_alloc_attrs.push_back(attr);
    }
    params.input_alloc_attrs = &input_alloc_attrs;
}

inline void TFContext::FillInputDeviceContext()
{
    input_device_contexts.clear();
    input_device_contexts.reserve(params.op_kernel->num_inputs());
    for (int index = 0; index < params.op_kernel->num_inputs(); index++) {
        input_device_contexts.push_back(nullptr);
    }
    params.input_device_contexts = &input_device_contexts;
}

void TFSession::registerContext(uint64_t taskId, TFContext *ctx)
{
    tensorflow::mutex_lock locker(m_muctx);
    auto res = m_contexts.insert({taskId, ctx});
    if (!res.second) {
        ERR("Register context failed. TaskId: {}", taskId);
    }
}

TFContext *TFSession::findContext(uint64_t taskId)
{
    tensorflow::mutex_lock locker(m_muctx);
    auto it = m_contexts.find(taskId);
    if (it != m_contexts.end()) {
        return it->second;
    }
    return nullptr;
}

void TFSession::deregisterContext(uint64_t taskId)
{
    tensorflow::mutex_lock locker(m_muctx);
    auto it = m_contexts.find(taskId);
    if (it != m_contexts.end()) {
        m_contexts.erase(it);
    } else {
        WARN("Deregistering non-exist context: {}", taskId);
    }
}

executor::TFOpContextUpdate TFSession::finalizeContext(TFContext *pContext)
{
    executor::TFOpContextUpdate tfctxupd;

    auto context = pContext->ctx();
    tfctxupd.set_status_code(context->status().code());
    tfctxupd.set_status_msg(context->status().error_message());
    tfctxupd.set_is_output_dead(*context->is_output_dead());

    // process tensor received by rendezvous
    // And this will clear tensors table in rendezvous
    for (auto &elem : pContext->rendez.releasePendingSentTensors()) {
        auto item = tfctxupd.add_rendeztensors();
        item->set_key(elem.first);
        item->set_allocattributes(elem.second.args.alloc_attrs.value);
        item->set_isdead(elem.second.isDead);
        tensorToProtoData(item->mutable_val(), &elem.second.val);
    }

    // process tensor set as outputs
    std::vector<std::string> output_names;
    for (int i = 0; i != context->num_outputs(); i++) {
        auto output_name = pContext->ctx()->op_kernel().name();
        if (i != 0) {
            tensorflow::strings::StrAppend(&output_name, ":", i);
        }
        output_names.push_back(output_name);
        INFO("Processing output: {}", output_name);

        auto out = context->release_output(i);
        // Let the session manage the tensor memory
        // The session takes the ownership of tensor in non-ref case
        registerTensorForName(output_name, out);

        auto outitem = tfctxupd.add_outputs();
        outitem->set_name(output_name);
        outitem->set_is_ref(out.is_ref());
        tensorToProtoMeta(outitem->mutable_meta(), out);
    }

    // Save tensors in TensorStore to session
    auto ok = pContext->tensorStore.SaveTensors(output_names, &m_sessState);
    if (!ok.ok()) {
        ERR("Error when save tensor store to session: {}", ok);
    }

    // Remove from context registary
    deregisterContext(pContext->seq);

    return tfctxupd;
}

