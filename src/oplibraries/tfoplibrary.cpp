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

#include "utils/protoutils.h"
#include "utils/pointerutils.h"
#include "platform/logging.h"

#include "executor.pb.h"
#include "tfoplibrary.pb.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace rpc = executor;
using ::tensorflow::NodeDef;
using ::google::protobuf::Message;
using std::unique_ptr;

bool TFOpLibrary::accepts(const rpc::OpKernelDef& operation)
{
    return operation.oplibrary() == rpc::OpKernelDef::TENSORFLOW;
}

unique_ptr<tensorflow::OpKernel> TFOpLibrary::kernelFromDef(const executor::OpKernelDef &opdef)
{
    auto msg = utils::createMessage("tensorflow.NodeDef", opdef.extra().data(), opdef.extra().size());
    if (!msg) {
        return {};
    }

    auto nodedef = utils::static_unique_ptr_cast<NodeDef>(std::move(msg));
    DEBUG("Got NodeDef {}", nodedef->DebugString());
    // FIXME: create opkernel from def
    return {};
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

TFTask::TFTask(TFOpLibrary *library, unique_ptr<tensorflow::OpKernel> &&kernel,
               unique_ptr<tensorflow::OpKernelContext> &&context)
    : m_opkernel(std::move(kernel))
    , m_context(std::move(context))
    , m_library(library)
{ }

rpc::ResultCode TFTask::run()
{
    if (m_opkernel && m_context) {
        m_opkernel->Compute(m_context.get());
    } else {
        ERR("Got nullptr for opkernel or context: m_opkernel = {:x}, m_context = {:x}",
            reinterpret_cast<uint64_t>(m_opkernel.get()), reinterpret_cast<uint64_t>(m_context.get()));
    }
}

rpc::OpContextDef TFTask::contextDef()
{
    return m_library->contextToDef(m_context.get());
}
