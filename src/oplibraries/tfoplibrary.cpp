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

#include "tfdevice.h"

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

unique_ptr<tensorflow::OpKernel> TFOpLibrary::kernelFromDef(const executor::OpKernelDef &opdef)
{
    google::protobuf::io::ArrayInputStream raw_input(opdef.extra().data(), opdef.extra().size());
    google::protobuf::io::CodedInputStream coded_input(&raw_input);

    auto nodedef = utils::createLenLimitedMessage<NodeDef>("tensorflow.NodeDef", &coded_input);
    if (!nodedef) { return {}; }
    DEBUG("Got NodeDef {}", nodedef->DebugString());

    auto configproto = utils::createLenLimitedMessage<ConfigProto>("tensorflow.ConfigProto", &coded_input);
    if (!configproto) { return {}; }
    DEBUG("Got ConfigProto {}", configproto->DebugString());

    auto funcdeflib = utils::createLenLimitedMessage<FunctionDefLibrary>("tensorflow.FunctionDefLibrary", &coded_input);
    if (!funcdeflib) { return {}; }
    DEBUG("Got funcdeflib {}", funcdeflib->DebugString());

    // TODO: compute session id
    std::string session_id = "session_id";

    auto sess = getOrCreateSession(session_id, opdef.graph_def_version(), *configproto, *funcdeflib);
    if (!sess) { return {}; }

    return sess->createKernel(*nodedef);
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


unique_ptr<tensorflow::OpKernelContext> TFOpLibrary::contextFromDef(const executor::OpContextDef &ctxdef)
{
    // FIXME: create kernel context from def
    return {};
}

executor::OpContextDef TFOpLibrary::contextToDef(const tensorflow::OpKernelContext *context)
{
    // FIXME: create def from kernel context
    return {};
}

ITask * TFOpLibrary::createTask(const rpc::OpKernelDef& opdef, const rpc::OpContextDef& ctxdef)
{
    return new TFTask(this, kernelFromDef(opdef), contextFromDef(ctxdef));
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
{ }

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

    return {};
}

