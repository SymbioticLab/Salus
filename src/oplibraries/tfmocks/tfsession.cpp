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

TFSession::TFSession(TFOpLibrary *opLibrary, const tensorflow::FunctionDefLibrary &fDefLib,
                     int graphDefVersion, const tensorflow::ConfigProto &configProto)
    : m_oplibrary(opLibrary)
    , m_sessHandle("executor_session")
    , m_opseg()
    , m_flibDef(tensorflow::OpRegistry::Global(), fDefLib)
    , m_fruntime(nullptr)
    , m_rendez(tensorflow::NewLocalRendezvous())
{
    DEBUG("Creating new TFSession at {:x}", reinterpret_cast<uint64_t>(this));

    m_options.config = configProto;

    m_device = std::make_unique<TFDevice>(m_options);

    m_fruntime.reset(tensorflow::NewFunctionLibraryRuntime(
        nullptr /* DeviceMgr */, m_options.env,
        m_device.get(), graphDefVersion, &m_flibDef, configProto.graph_options().optimizer_options()));

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
    m_rendez->Unref();
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

tensorflow::OpKernel *TFSession::findOrCreateKernel(const tensorflow::NodeDef &ndef)
{
    tensorflow::OpKernel *kernel = nullptr;
    // Caches the kernel only if the node is stateful.
    if (!m_fruntime->IsStateful(ndef.op())) {
        auto ok = m_fruntime->CreateKernel(ndef, &kernel);
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
    auto lib = m_fruntime.get();
    auto create_fn = [lib, &ndef](tensorflow::OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
    };
    auto ok = m_opseg.FindOrCreate(m_sessHandle, ndef.name(), &kernel, create_fn);
    if (!ok.ok()) {
        ERR("Failed to create kernel with status {} for NodeDef: {}", ok,
            ndef.DebugString());
    }

    return kernel;
}

TFContext::TFContext(TFSession *sess, uint64_t taskId)
    : step_container(0, [](const std::string&) {})
    , rendez(sess)
    , m_taskId(taskId)
    , m_sess(sess)
{
}

TFContext::~TFContext() {
    m_sess->contextDestroied(m_taskId);
}

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

TFContext *TFSession::findContext(uint64_t taskId)
{
    tensorflow::mutex_lock locker(m_muctx);
    auto it = m_contexts.find(taskId);
    if (it != m_contexts.end()) {
        return it->second;
    }
    return nullptr;
}

void TFSession::contextDestroied(uint64_t taskId)
{
    tensorflow::mutex_lock locker(m_muctx);
    m_contexts.erase(taskId);
}

std::unique_ptr<TFContext> TFSession::createContext(const executor::TFOpContextDef &tfdef,
                                                    tensorflow::OpKernel *opkernel, uint64_t taskId)
{
    auto tfctx = std::make_unique<TFContext>(this, taskId);
    {
        tensorflow::mutex_lock locker(m_muctx);
        m_contexts[taskId] = tfctx.get();
    }

    tfctx->params.device = m_device.get();
    tfctx->params.op_kernel = opkernel;
    tfctx->params.step_container = &tfctx->step_container;
    tfctx->params.slice_reader_cache = &tfctx->slice_reader_cache_wrapper;
    tfctx->params.resource_manager = m_device->resource_manager();
    tfctx->params.function_library = m_fruntime.get();
    tfctx->params.rendezvous = &tfctx->rendez;

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
