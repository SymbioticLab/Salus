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

#ifndef TFSESSION_H
#define TFSESSION_H

#include "tfrendezvous.h"

#include "executor.pb.h"
#include "tfoplibrary.pb.h"

#include <tensorflow/core/framework/op_segment.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/util/tensor_slice_reader_cache.h>
#include <tensorflow/core/platform/mutex.h>
#include <tensorflow/core/public/session_options.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using tensorflow::TensorValue;
typedef tensorflow::gtl::InlinedVector<tensorflow::TensorValue, 4> TensorValueVec;
typedef tensorflow::gtl::InlinedVector<tensorflow::DeviceContext*, 4> DeviceContextVec;
typedef tensorflow::gtl::InlinedVector<tensorflow::AllocatorAttributes, 4> AllocatorAttributeVec;

namespace tensorflow {
class OptimizerOptions;
class NodeDef;
class FunctionDefLibrary;
class ConfigProto;
}

class TFDevice;
class TFSession;
class TFOpLibrary;

class TFContext
{
public:
    TFContext(TFSession *sess, uint64_t taskId);
    ~TFContext();

    tensorflow::OpKernelContext *ctx();

    void FillOutputAttrs();
    void FillInputAttrs();
    void FillInputDeviceContext();

    tensorflow::ScopedStepContainer step_container;
    tensorflow::checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    TFRendezvous rendez;

    TensorValueVec inputs;

    DeviceContextVec input_device_contexts;
    AllocatorAttributeVec input_alloc_attrs;

    std::vector<tensorflow::AllocatorAttributes> output_attrs;

    tensorflow::OpKernelContext::Params params;
private:
    std::unique_ptr<tensorflow::OpKernelContext> context;
    uint64_t m_seq;
    TFSession *m_sess;
};

class MaybeLock
{
public:
    explicit MaybeLock(TensorValue &val) : m_val(val)
    {
        if (m_val.is_ref()) {
            m_val.mutex_if_ref->lock();
        }
    }
    ~MaybeLock()
    {
        if (m_val.is_ref()) {
            m_val.mutex_if_ref->unlock();
        }
    }
private:
    TensorValue &m_val;
};

class TFSession
{
public:
    TFSession(TFOpLibrary *opLibrary, const tensorflow::FunctionDefLibrary &fDefLib,
              int graphDefVersion, const tensorflow::ConfigProto &configProto);

    ~TFSession();

    tensorflow::OpKernel *findOrCreateKernel(const tensorflow::NodeDef &nodedef);

    std::unique_ptr<TFContext> createContext(const executor::TFOpContextDef &tfdef,
                                             tensorflow::OpKernel *opkernel, uint64_t taskId);
    TFContext *findContext(uint64_t taskId);

    // Tensor memory management
    bool findTensorFromName(const std::string &name, TensorValue &val);
    void registerTensorForName(const std::string &name, TensorValue val);
    /**
     * Create a tensor from proto, allocate and fill in memory,
     */
    std::unique_ptr<tensorflow::Tensor> tensorFromProtoData(const tensorflow::TensorProto &proto);

    void tensorToProtoMeta(tensorflow::TensorProto *meta, TensorValue val);
    void tensorToProtoData(tensorflow::TensorProto *data, TensorValue val);

    bool isCompatible(const tensorflow::Tensor &tensor, const tensorflow::TensorProto &proto) const;

private:
    friend class TFContext;
    void contextDestroied(uint64_t seq);

    tensorflow::SessionOptions m_options;

    TFOpLibrary *m_oplibrary;

    std::string m_sessHandle;

    tensorflow::OpSegment m_opseg;
    std::vector<std::unique_ptr<tensorflow::OpKernel>> m_kernels;

    tensorflow::FunctionLibraryDefinition m_flibDef;
    std::unique_ptr<tensorflow::FunctionLibraryRuntime> m_fruntime;
    std::unique_ptr<TFDevice> m_device;

    friend class TFRendezvous;
    tensorflow::Rendezvous *m_rendez;

    tensorflow::mutex m_mu;
    std::unordered_map<std::string, TensorValue> m_tensors;

    tensorflow::mutex m_muctx;
    std::unordered_map<uint64_t, TFContext*> m_contexts;
};

#endif // TFSESSION_H
