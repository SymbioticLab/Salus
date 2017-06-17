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
    , m_rendez(this)
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
        delete p.second;
        MemoryMgr::instance().deallocate(reinterpret_cast<void*>(p.first));
    }
}

void TFSession::registerTensorMemory(const tensorflow::Tensor &tensor)
{
    if (!tensor.IsInitialized()) {
        INFO("Skipped registering uninitialized tensor: {}", tensor.DebugString());
        return;
    }
    registerTensorMemoryLocked(new tensorflow::Tensor(tensor));
}

tensorflow::Tensor *TFSession::createAndRegister(const tensorflow::TensorProto &proto)
{
    if (!m_device) {
        ERR("m_device should not be nullptr for TFSession");
        return nullptr;
    }

    auto tensor = new tensorflow::Tensor;

    auto status = m_device->MakeTensorFromProto(proto, {}, tensor);
    if (!status.ok()) {
        ERR("Error when create tensor");
        return nullptr;
    }

    registerTensorMemoryLocked(tensor);
    return tensor;
}

void TFSession::registerTensorMemoryLocked(tensorflow::Tensor *tensor)
{
    if (!tensor) {
        ERR("Got nullptr in registerTensorMemoryLocked");
        return;
    }
    INFO("Registering tensor: {}", tensor->DebugString());
    tensorflow::mutex_lock locker(m_mu);
    m_tensors[reinterpret_cast<uint64_t>(tensor->tensor_data().data())] = tensor;
}

tensorflow::Tensor *TFSession::tensorFromAddrHandle(uint64_t addr_handle)
{
    tensorflow::mutex_lock locker(m_mu);
    if (m_tensors.count(addr_handle) <= 0) {
        ERR("Tensor at addr {:x} not found", addr_handle);
        return {};
    }
    return m_tensors.at(addr_handle);
}

tensorflow::Tensor *TFSession::findTensorFromProtoMeta(const tensorflow::TensorProto &proto)
{
    auto addr = proto.int64_val(0);
    auto tensor = tensorFromAddrHandle(addr);
    if (!tensor) {
        ERR("Requested tensor not found at addr {:x}", addr);
        return {};
    }

    if (!isCompatible(*tensor, proto)) {
        return {};
    }
    return tensor;
}

bool TFSession::isCompatible(const tensorflow::Tensor &tensor, const tensorflow::TensorProto &proto) const
{
    tensorflow::TensorShape shape(proto.tensor_shape());
    if (tensor.dtype() != proto.dtype()
        || tensor.shape() != shape) {
        ERR("Requested tensor metadata mismatch with record. Requested: {} of type {}, stored: {} of type {}",
            tensor.shape().DebugString(), tensor.dtype(),
            shape.DebugString(), proto.dtype());
        return false;
    }
    return true;
}

void TFSession::tensorMetaToProto(tensorflow::TensorProto *proto, const tensorflow::Tensor &tensor)
{
    proto->set_dtype(tensor.dtype());
    tensor.shape().AsProto(proto->mutable_tensor_shape());

    auto addr_handle = reinterpret_cast<uint64_t>(tensor.tensor_data().data());
    // HACK: use a int64 val entry to store the addr handle for simplicity,
    // idealy should store this in tensor_content with proper encoding.
    proto->add_int64_val(addr_handle);
}

TFRendezvous::SentTensorTable TFSession::releasePendingRendezSentTensors()
{
    return m_rendez.releasePendingSentTensors();
}

TFRendezvous::RecvTable TFSession::releasePendingRendezRecv()
{
    return m_rendez.releasePendingRecv();
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

TFContext::TFContext(TFSession *sess)
    : step_container(0, [](const std::string&) {})
{
    UNUSED(sess);
}

TFContext::~TFContext() {
    for (auto t : inputs) {
        delete t.tensor;
    }
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

std::unique_ptr<TFContext> TFSession::createContext(const executor::TFOpContextDef &tfdef,
                                                    tensorflow::OpKernel *opkernel)
{
    auto tfctx = std::make_unique<TFContext>(this);

    tfctx->params.device = m_device.get();
    tfctx->params.op_kernel = opkernel;
    tfctx->params.step_container = &tfctx->step_container;
    tfctx->params.slice_reader_cache = &tfctx->slice_reader_cache_wrapper;
    tfctx->params.resource_manager = m_device->resource_manager();
    tfctx->params.function_library = m_fruntime.get();
    tfctx->params.rendezvous = &m_rendez;

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
    auto input_types = opkernel->input_types();
    for (const auto &inpdef : tfdef.inputs()) {
        auto input = findTensorFromProtoMeta(inpdef);
        auto pmu = IsRefType(input_types[tfctx->inputs.size()]) ? &tfctx->ref_mutex : nullptr;
        tfctx->inputs.push_back({pmu, input});
    }
    tfctx->params.inputs = &tfctx->inputs;

    return tfctx;
}
