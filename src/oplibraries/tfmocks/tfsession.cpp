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
#include "tfallocator.h"
#include "tfdevice.h"
#include "tfrendezvous.h"

#include "memorymgr/memorymgr.h"
#include "oplibraries/tfoplibrary.h"
#include "platform/logging.h"
#include "utils/macros.h"
#include "utils/threadutils.h"

#include "tfoplibrary.pb.h"

#include <tensorflow/core/common_runtime/copy_tensor.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/common_runtime/function.h>
#include <tensorflow/core/common_runtime/rpc_device/exec_helpers/exechelpers.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/graph/graph.h>
#include <tensorflow/core/graph/graph_constructor.h>
#include <tensorflow/core/lib/gtl/stl_util.h>
#include <tensorflow/core/protobuf/config.pb.h>

using namespace tensorflow::remote;

TFSession::TFSession(TFOpLibrary *opLibrary, const std::string &sessionId,
                     const tensorflow::ConfigProto &configProto)
    : m_oplibrary(opLibrary)
    , m_sessionId(sessionId)
    , m_options()
    , m_cpuAllocator(std::make_unique<TFAllocator>())
    , m_opseg()
{
    m_options.config = configProto;

    m_opseg.AddHold(m_sessionId);

    DEBUG("TFSession created at {:x}, sessionId: {}", reinterpret_cast<uint64_t>(this), m_sessionId);
}

bool TFSession::initialize()
{
    return true;
}

TFSession::~TFSession()
{
    m_opseg.RemoveHold(m_sessionId);
    for (auto p : m_tensors) {
        if (!p.second.val.is_ref()) {
            delete p.second.val.tensor;
        }
        if (p.second.context) {
            p.second.context->Unref();
        }
    }
}

bool TFSession::findTensorFromName(const std::string &name, TensorItem &item)
{
    tensorflow::mutex_lock locker(m_mu);
    auto it = m_tensors.find(name);
    if (it == m_tensors.end()) {
        ERR("Tensor not found under name: {}", name);
        return false;
    }
    item = it->second;
    return true;
}

void TFSession::registerTensorForName(const std::string &name, TensorItem item)
{
    INFO("Registering tensor: {}, is ref: {}, under name: {}, buffer: {:x}", item.val->shape().DebugString(),
         item.val.is_ref(), name, reinterpret_cast<uint64_t>(item.val->tensor_data().data()));

    if (item.context) {
        item.context->Ref();
    }

    tensorflow::mutex_lock locker(m_mu);
    auto it = m_tensors.find(name);
    if (it == m_tensors.end()) {
        m_tensors.emplace(name, item);
    } else {
        if (it->second.val.mutex_if_ref != item.val.mutex_if_ref) {
            WARN("The tensor going to be registered already exists, and is under a different mutex");
        }
        it->second = item;
    }
}

bool TFSession::isCompatible(const tensorflow::Tensor &tensor, const tensorflow::TensorProto &proto) const
{
    auto dtype = proto.dtype();
    if (tensorflow::IsRefType(dtype)) {
        dtype = tensorflow::RemoveRefType(proto.dtype());
    }
    tensorflow::TensorShape shape(proto.tensor_shape());
    if (tensor.dtype() != dtype || tensor.shape() != shape) {
        ERR("Requested tensor metadata mismatch with record. Requested: {} of type {}, stored: {} "
            "of type {}",
            tensor.shape().DebugString(), tensor.dtype(), shape.DebugString(), proto.dtype());
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
        DEBUG("Reuse existing Execution {} in session {}", execId, m_sessionId);
        return e;
    }

    auto execState = std::make_unique<TFExecutionState>(this, execId, std::move(graphdef),
                                                        m_options.config.graph_options().optimizer_options());
    if (!execState->initialize()) {
        ERR("Failed to initialize execution state");
        execState.reset();
        return nullptr;
    }

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
    return nullptr;
}

tensorflow::Device *TFSession::findDevice(const DeviceSpec &dev) const
{
    std::string name;
    switch (dev.type) {
    case DeviceType::CPU:
        name = "CPU:";
        break;
    case DeviceType::GPU:
        name = "GPU:";
        break;
    default:
        name = "CPU:";
        break;
    }
    name += std::to_string(dev.id);

    tensorflow::Device *device = nullptr;
    auto ok = m_oplibrary->deviceManager()->LookupDevice(name, &device);
    if (!ok.ok()) {
        ERR("Cannot find device for {}: {}", dev, ok);
    }
    return device;
}

TFExecutionState::TFExecutionState(TFSession *sess, const std::string &execId, tensorflow::GraphDef &&graph,
                                   const tensorflow::OptimizerOptions &optOptions)
    : m_session(sess)
    , m_execId(execId)
    , m_graphdef(std::move(graph))
    , m_fdefinition(nullptr)
    , m_graph(nullptr)
    , m_optOptions(optOptions)
    , m_rendez(tensorflow::NewLocalRendezvous())
{
    m_fdefinition = std::make_unique<tensorflow::FunctionLibraryDefinition>(tensorflow::OpRegistry::Global(),
                                                                            m_graphdef.library());
}

bool TFExecutionState::initialize()
{
    m_graph = ExecHelpers::convertGraphDefToGraph(m_graphdef, m_fdefinition.get(), m_gindex);
    if (!m_graph) {
        ERR("Create graph from graphdef failed");
        return false;
    }

    m_gview = std::make_unique<GraphView>();
    m_gview->Initialize(m_graph.get());

    logging::logger()->flush();
    for (auto &elem : m_gindex) {
        INFO("GIndex: {} -> {}", elem.first, elem.second);
    }
    return true;
}

const std::string &TFExecutionState::execId() const
{
    return m_execId;
}

TFSession *TFExecutionState::session() const
{
    return m_session;
}

tensorflow::FunctionLibraryRuntime *TFExecutionState::functionRuntime(tensorflow::Device *tfdev)
{
    tensorflow::mutex_lock l(m_mu);

    auto &ptr = m_fruntimes[tfdev];

    if (!ptr) {
        ptr.reset(tensorflow::NewFunctionLibraryRuntime(nullptr, m_session->sessionOptions().env, tfdev,
                                                        m_graphdef.versions().producer(), m_fdefinition.get(),
                                                        m_optOptions));
    }
    return ptr.get();
}

tensorflow::DeviceContext *TFExecutionState::deviceContext(const std::string &name, tensorflow::Device *tfdev)
{
    auto nit = m_gindex.find(name);
    if (nit == m_gindex.end()) {
        ERR("{} not found in graph", name);
        return nullptr;
    }
    uint32_t nid = nit->second;

    auto it = m_deviceContexts.end();
    {
        tensorflow::mutex_lock l(m_mu);

        it = m_deviceContexts.find(tfdev);
        if (it == m_deviceContexts.end()) {
            tensorflow::DeviceContextMap contexts;
            auto ok = tfdev->FillContextMap(m_graph.get(), &contexts);
            if (!ok.ok()) {
                ERR("filling contextmap failed: {}", ok);
            }
            std::tie(it, std::ignore) = m_deviceContexts.emplace(tfdev, std::move(contexts));
        }
    }
    if (it != m_deviceContexts.end() && nid < it->second.size()) {
        return it->second[nid];
    }
    return nullptr;
}

tensorflow::Rendezvous *TFExecutionState::rendez() const
{
    return m_rendez;
}

tensorflow::Graph *TFExecutionState::graph() const
{
    return m_graph.get();
}

GraphView *TFExecutionState::gview() const
{
    return m_gview.get();
}

tensorflow::Node *TFExecutionState::findNodeInGraph(const std::string &name) const
{
    auto nit = m_gindex.find(name);
    if (nit == m_gindex.end()) {
        ERR("{} not found in graph", name);
        return nullptr;
    }
    uint32_t nid = nit->second;
    return m_graph->FindNodeId(nid);
}

TFExecutionState::~TFExecutionState()
{
    m_rendez->Unref();
    for (auto &item : m_deviceContexts) {
        for (auto ctx : item.second) {
            ctx->Unref();
        }
    }
}

bool TFSession::findOrCreateKernel(TFExecutionState *execState, const tensorflow::NodeDef &ndef,
                                   tensorflow::OpKernel *&kernel, DeviceSpec &dev)
{
    kernel = nullptr;

    // First check if we already created the kernel on some device
    bool found = true;
    auto ok = m_opseg.FindOrCreate(m_sessionId, ndef.name(), &kernel, [&found](auto) {
        found = false;
        return tensorflow::Status::OK();
    });
    if (ok.ok() && found) {
        // we saw this kernel before, check if the device match
        auto it = m_kernelDevice.end();
        {
            tensorflow::mutex_lock l(m_muk);
            it = m_kernelDevice.find(kernel);
            if (it == m_kernelDevice.end()) {
                ERR("We've created the kernel, but don't remember its device: {}", ndef);
                kernel = nullptr;
                return false;
            }
        }
        if (dev == it->second) {
            // We are on the same device, good.
            return true;
        }
        ERR("Stateful kernel can not be moved: previously created on {}, now requested on {}", it->second,
            dev);
        return false;
    } else if (!ok.ok()) {
        ERR("Failed to create kernel with status {} for NodeDef: {}", ok, ndef);
        // continue to create the kernel
    }

    auto tfdev = findDevice(dev);
    if (!tfdev) {
        ERR("Cannot find suitable device for spec: {}", dev);
        return false;
    }
    INFO("Creating a kernel for device: {}", tfdev->name());

    // Caches the kernel only if the node is stateful.
    auto fruntime = execState->functionRuntime(tfdev);
    if (!fruntime->IsStateful(ndef.op())) {
        auto ok = fruntime->CreateKernel(ndef, &kernel);
        if (!ok.ok()) {
            ERR("Failed to create kernel with status {} for NodeDef: {}", ok, ndef);
            delete kernel;
            kernel = nullptr;
            return false;
        }
        tensorflow::mutex_lock l(m_muk);
        // TODO: kernel created in this way can be deleted after tfctx is done
        m_kernels.emplace_back(kernel);
        return true;
    }

    // Kernels created for subgraph nodes need to be cached.  On
    // cache miss, create_fn() is invoked to create a kernel based
    // on the function library here + global op registry.
    // OpSegment takes ownership of the created kernel.
    auto create_fn = [fruntime, &ndef, &dev, this](tensorflow::OpKernel **kernel) {
        auto ok = fruntime->CreateKernel(ndef, kernel);
        if (ok.ok()) {
            tensorflow::mutex_lock l(m_muk);
            m_kernelDevice[*kernel] = dev;
        }
        return ok;
    };
    ok = m_opseg.FindOrCreate(m_sessionId, ndef.name(), &kernel, create_fn);
    if (!ok.ok()) {
        ERR("Failed to create kernel with status {} for NodeDef: {}", ok, ndef);
        kernel = nullptr;
        return false;
    }

    return true;
}

NodeItem *TFExecutionState::prepareNodeItemOnDevice(tensorflow::OpKernel *opkernel, tensorflow::Device *d)
{
    assert(opkernel);
    assert(d);

    auto node = findNodeInGraph(opkernel->name());
    if (!node) {
        return nullptr;
    }

    auto nodeItem = m_gview->node(node->id());
    nodeItem->kernel = opkernel;
    auto ok = m_gview->SetAllocAttrForNode(node, d);
    if (!ok.ok()) {
        ERR("Infering alloc attr for node {} failed: {}", opkernel->name(), ok);
        return nullptr;
    }

    nodeItem->kernel_is_expensive = nodeItem->kernel->IsExpensive();
    nodeItem->kernel_is_async = (nodeItem->kernel->AsAsync() != nullptr);
    nodeItem->is_merge = tensorflow::IsMerge(node);
    nodeItem->is_enter = tensorflow::IsEnter(node);
    nodeItem->is_exit = tensorflow::IsExit(node);
    nodeItem->is_control_trigger = tensorflow::IsControlTrigger(node);
    nodeItem->is_sink = tensorflow::IsSink(node);
    nodeItem->is_enter_exit_or_next_iter =
        (tensorflow::IsEnter(node) || tensorflow::IsExit(node) || tensorflow::IsNextIteration(node));

    return nodeItem;
}

std::unique_ptr<TFContext> TFSession::createContext(const executor::TFOpContextDef &tfdef,
                                                    tensorflow::OpKernel *opkernel, uint64_t seq,
                                                    TFExecutionState *execState, const DeviceSpec &dev)
{
    auto tfctx = std::make_unique<TFContext>(execState, seq);
    registerContext(seq, tfctx.get());

    auto device = findDevice(dev);

    tfctx->node_item = execState->prepareNodeItemOnDevice(opkernel, device);

    tfctx->params.device = device;
    tfctx->params.op_kernel = opkernel;
    tfctx->params.op_device_context = execState->deviceContext(opkernel->name(), device);
    tfctx->params.resource_manager = device->resource_manager();
    tfctx->params.function_library = execState->functionRuntime(device);

    tfctx->params.step_id = tfdef.step_id();
    tfctx->params.frame_iter = tensorflow::FrameAndIter(tfdef.frame_id(), tfdef.iter_id());
    tfctx->params.is_input_dead = tfdef.is_input_dead();

    tfctx->params.output_attr_array = tfctx->node_item->output_attrs();
    //     tfctx->FillOutputAttrs(tfdef);

    auto num_inputs = opkernel->num_inputs();
    if (num_inputs != tfdef.inputs_size()) {
        ERR("Missing inputs in received TFOpContextDef: required {}, found {}", num_inputs,
            tfdef.inputs_size());
        return {};
    }
    tfctx->inputs.reserve(num_inputs);
    tfctx->input_device_contexts.reserve(num_inputs);
    tfctx->input_alloc_attrs.reserve(num_inputs);
    for (int i = 0; i != num_inputs; ++i) {
        const auto &initem = tfdef.inputs(i);
        if (initem.name() != opkernel->def().input(i)) {
            ERR("Mismatch input: {}, expected: {}", initem.name(), opkernel->def().input(i));
            return {};
        }
        TensorItem input;
        if (!findTensorFromName(initem.name(), input)) {
            ERR("Input not found");
            return {};
        }

        auto inattrs = input.nodeItem->output_attrs()[input.slot];

        // Handle every combination of input and op types
        // ----------------------------------------------
        //    Input   |   Op   |   Device   |   Result   |
        //     ref       noref      same         deref
        //    noref      noref      same        nothing
        //     ref        ref       same        nothing
        //    noref       ref       same         reject
        //     ref       noref      diff        devcopy
        //    noref      noref      diff        devcopy
        //     ref        ref       diff         reject
        //    noref       ref       diff         reject

        if (initem.is_ref() && !input.val.is_ref()) {
            ERR("{}-th input expects a ref type", i);
            return {};
        } else if (initem.is_ref() && input.val.is_ref()) {
            if (device != input.device) {
                ERR("Operation {} expects an reference, but input[{}] {} is on different device.",
                    opkernel->name(), i, initem.name());
                return {};
            }
        } else if (!initem.is_ref()) {
            if (device != input.device) {
                // Operation and input on different device,
                // do a copy tensor to ensure input tensor is on the same device
                tfctx->deref_tensors.emplace_back(device->GetAllocator({}), input.val->dtype(),
                                                  input.val->shape());
                auto &copy = tfctx->deref_tensors.back();

                tensorflow::Notification n;
                tensorflow::Status ok;
                tensorflow::CopyTensor::ViaDMA(initem.name(), input.context, tfctx->params.op_device_context,
                                               input.device, device, inattrs, {}, input.val.tensor, &copy,
                                               [&n, &ok](auto status) {
                                                   ok = status;
                                                   n.Notify();
                                               });
                n.WaitForNotification();

                if (!ok.ok()) {
                    ERR("Copying from device {} to device {} failed when preparing {}-th input {} "
                        "for op {}: {}",
                        input.device->name(), device->name(), i, initem.name(), opkernel->name(), ok);
                }

                input.context = tfctx->params.op_device_context;
                input.device = device;
                input.val = {nullptr, &copy};
                inattrs = {};
            } else if (input.val.is_ref()) {
                // Automatically deref the tensor ref when the op expects a
                // tensor but is given a ref to a tensor.  Need to deref it
                // under the mutex.
                MaybeLock l(input.val);
                tfctx->deref_tensors.emplace_back(*input.val.tensor);
                auto &t = tfctx->deref_tensors.back();
                input.val = {nullptr, &t};
            }
        }

        tfctx->input_alloc_attrs.push_back(inattrs);
        tfctx->input_device_contexts.push_back(input.context);
        tfctx->inputs.push_back(input.val);
    }

    return tfctx;
}

TFContext::TFContext(TFExecutionState *exec, uint64_t seq)
    : seq(seq)
    , step_container(0, [](const std::string &) {})
    , rendez(exec)
    , m_exec(exec)
{
    params.step_container = &step_container;
    params.slice_reader_cache = &slice_reader_cache_wrapper;
    params.rendezvous = &rendez;
    params.tensor_store = &tensorStore;
    params.input_alloc_attrs = &input_alloc_attrs;
    params.input_device_contexts = &input_device_contexts;
    params.inputs = &inputs;
}

TFContext::~TFContext() = default;

tensorflow::OpKernelContext *TFContext::ctx()
{
    if (!context) {
        context.reset(new tensorflow::OpKernelContext(&params));
    }
    return context.get();
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
    auto pending = pContext->rendez.releasePendingSentTensors();
    utils::semaphore se;
    for (auto &elem : pending) {
        auto item = tfctxupd.add_rendeztensors();
        item->set_key(elem.first);
        item->set_allocattributes(elem.second.args.alloc_attrs.value);
        item->set_isdead(elem.second.isDead);

        auto &val = elem.second.val;
        auto devCtx = elem.second.args.device_context;
        auto mval = item->mutable_val();
        if (devCtx) {
            if (!elem.second.args.alloc_attrs.on_host()) {
                tensorflow::Tensor cputensor(m_cpuAllocator.get(), val.dtype(), val.shape());
                auto dev = static_cast<tensorflow::Device *>(context->device());
                devCtx->CopyDeviceTensorToCPU(&val, elem.first, dev, &cputensor,
                                              [&se, &mval, devCtx, copy = cputensor, this ](auto) mutable {
                                                  this->tensorToProtoData(mval, &copy);
                                                  se.notify();
                                                  devCtx->Unref();
                                              });
            } else {
                tensorToProtoData(mval, &val);
                se.notify();
            }
        } else {
            WARN("Device context is nullptr, assuming CPU device. The tensor not copied from "
                 "device to cpu.");
            tensorToProtoData(mval, &val);
            se.notify();
        }
    }
    se.wait(pending.size());

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
        registerTensorForName(output_name,
                              {out, context->op_device_context(),
                               static_cast<tensorflow::Device *>(context->device()), pContext->node_item, i});

        auto outitem = tfctxupd.add_outputs();
        outitem->set_name(output_name);
        outitem->set_is_ref(out.is_ref());
        auto mval = outitem->mutable_meta();
        auto devCtx = context->op_device_context();
        if (!context->output_alloc_attr(i).on_host() || !devCtx) {
            tensorToProtoMeta(mval, out);
        } else {
            // FIXME: proper handle in tensorflow required.
            tensorToProtoMeta(mval, out);
            /*
            MaybeLock l(out);
            tensorflow::Tensor copy(m_allocator.get(), out->dtype(), out->shape());
            auto dev = static_cast<tensorflow::Device*>(context->device());
            tensorflow::Notification n;
            devCtx->CopyDeviceTensorToCPU(out.tensor, output_name, dev, &copy,
                                        [&n](auto) mutable {
                n.Notify();
            });
            n.WaitForNotification();
            tensorToProtoData(mval, &copy);
            */
        }
    }

    // Save tensors in context's TensorStore to session
    auto ok = pContext->tensorStore.SaveTensors(output_names, &m_sessState);
    if (!ok.ok()) {
        ERR("Error when save tensor store to session: {}", ok);
    }

    // Remove from context registary
    deregisterContext(pContext->seq);

    return tfctxupd;
}
