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

#ifndef TFOPLIBRARY_H
#define TFOPLIBRARY_H

#include "ioplibrary.h"

#include "executor.pb.h"
#include "tfoplibrary.pb.h"

#include <tensorflow/core/framework/op_segment.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/util/tensor_slice_reader_cache.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorflow {
class OpSegment;
class OptimizerOptions;
class NodeDef;
class FunctionDefLibrary;
class ConfigProto;
}

class TFOpLibrary;
class TFDevice;
class TFSession;

typedef tensorflow::gtl::InlinedVector<tensorflow::TensorValue, 4> TensorValueVec;
typedef tensorflow::gtl::InlinedVector<tensorflow::DeviceContext*, 4> DeviceContextVec;
typedef tensorflow::gtl::InlinedVector<tensorflow::AllocatorAttributes, 4> AllocatorAttributeVec;

class TFContext
{
public:
    TFContext();
    ~TFContext();

    tensorflow::OpKernelContext *ctx();

    void FillOutputAttrs();

    tensorflow::ScopedStepContainer step_container;
    tensorflow::checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;

    TensorValueVec inputs;
    DeviceContextVec input_device_contexts;
    AllocatorAttributeVec input_alloc_attrs;

    std::vector<tensorflow::AllocatorAttributes> output_attrs;

    tensorflow::OpKernelContext::Params params;
private:
    std::unique_ptr<tensorflow::OpKernelContext> context;
};

class TFTask : public ITask
{
public:
    ~TFTask() override = default;

    TFTask(TFOpLibrary *library, std::unique_ptr<tensorflow::OpKernel> &&kernel,
           std::unique_ptr<TFContext> &&context);

    executor::Status run() override;
    executor::OpContextDef contextDef() override;

private:
    std::unique_ptr<tensorflow::OpKernel> m_opkernel;
    std::unique_ptr<TFContext> m_context;
    TFOpLibrary *m_library;
};

/**
 * @todo write docs
 */
class TFOpLibrary : public IOpLibrary
{
public:
    ~TFOpLibrary() override = default;

    bool accepts(const executor::OpKernelDef &operation) override;
    std::unique_ptr<ITask> createTask(const executor::OpKernelDef &opdef,
                                      const executor::OpContextDef &ctxdef) override;

    executor::OpContextDef contextToDef(tensorflow::OpKernelContext *context);

private:
    TFSession *getOrCreateSession(const std::string &sess_id, int graph_def_version,
                                  const tensorflow::ConfigProto &cfgProto,
                                  const tensorflow::FunctionDefLibrary &fDefLib);
    TFSession *getSession(const std::string &sess_id);

    std::mutex m_mu; // protects m_sessions
    std::unordered_map<std::string, std::unique_ptr<TFSession>> m_sessions;
};

class TFSession
{
public:
    TFSession(TFOpLibrary *opLibrary, const tensorflow::FunctionDefLibrary &fDefLib,
              int graphDefVersion, const tensorflow::OptimizerOptions &optimizerOpts);

    ~TFSession();

    std::unique_ptr<tensorflow::OpKernel> createKernel(const tensorflow::NodeDef &nodedef);

    std::unique_ptr<TFContext> createContext(const executor::TFOpContextDef &tfdef, tensorflow::OpKernel *opkernel);

private:
    TFOpLibrary *m_oplibrary;

    tensorflow::OpSegment m_opseg;
    tensorflow::FunctionLibraryDefinition m_flibDef;
    std::unique_ptr<tensorflow::FunctionLibraryRuntime> m_fruntime;
    std::unique_ptr<TFDevice> m_device;
};
#endif // TFOPLIBRARY_H
