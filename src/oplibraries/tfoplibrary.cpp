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

#include "executor.pb.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace rpc = executor;

bool TFOpLibrary::accepts(const rpc::OpKernelDef& operation)
{
    return operation.oplibrary() == rpc::OpKernelDef::TENSORFLOW;
}

tensorflow::OpKernel *TFOpLibrary::kernelFromDef(const executor::OpKernelDef &opdef)
{
    // FIXME: create opkernel from def
}

tensorflow::OpKernelContext *TFOpLibrary::contextFromDef(const executor::OpContextDef &ctxdef)
{
    // FIXME: create kernel context from def
}

executor::OpContextDef TFOpLibrary::contextToDef(const tensorflow::OpKernelContext *context)
{
    // FIXME: create def from kernel context
}

ITask * TFOpLibrary::createTask(const rpc::OpKernelDef& opdef, const rpc::OpContextDef& ctxdef)
{
    tensorflow::OpKernel *kernel = kernelFromDef(opdef);
    tensorflow::OpKernelContext *context = contextFromDef(ctxdef);
    return new TFTask(this, kernel, context);
}

TFTask::TFTask(TFOpLibrary *library, tensorflow::OpKernel *kernel, tensorflow::OpKernelContext *context)
    : m_opkernel(kernel)
    , m_context(context)
    , m_library(library)
{ }

rpc::ResultCode TFTask::run()
{
    m_opkernel->Compute(m_context.get());
}

rpc::OpContextDef TFTask::contextDef()
{
    return m_library->contextToDef(m_context.get());
}
