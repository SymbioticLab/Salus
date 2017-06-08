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
    explicit TFContext(TFSession *sess);
    ~TFContext();

    tensorflow::OpKernelContext *ctx();

    void FillOutputAttrs();
    void FillInputAttrs();
    void FillInputDeviceContext();

    tensorflow::ScopedStepContainer step_container;
    tensorflow::checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;

    TensorValueVec inputs;

    DeviceContextVec input_device_contexts;
    AllocatorAttributeVec input_alloc_attrs;
    tensorflow::mutex ref_mutex;

    std::vector<tensorflow::AllocatorAttributes> output_attrs;

    TFRendezvous rendez;

    tensorflow::OpKernelContext::Params params;
private:
    std::unique_ptr<tensorflow::OpKernelContext> context;
};

class TFSession
{
public:
    TFSession(TFOpLibrary *opLibrary, const tensorflow::FunctionDefLibrary &fDefLib,
              int graphDefVersion, const tensorflow::ConfigProto &configProto);

    ~TFSession();

    tensorflow::OpKernel *findOrCreateKernel(const tensorflow::NodeDef &nodedef);

    std::unique_ptr<TFContext> createContext(const executor::TFOpContextDef &tfdef, tensorflow::OpKernel *opkernel);

    // Tensor memory management
    void registerTensorMemory(const tensorflow::Tensor &tensor);
    tensorflow::Tensor *tensorFromAddrHandle(uint64_t addr_handle);
    tensorflow::Tensor *findTensorFromProtoMeta(const tensorflow::TensorProto &proto);

    /**
     * Convinence method that combines create a tensor from proto, allocate and fill in memory,
     * and finally register
     */
    tensorflow::Tensor *createAndRegister(const tensorflow::TensorProto &proto);

    void tensorMetaToProto(tensorflow::TensorProto *proto, const tensorflow::Tensor &tensor);

    bool isCompatible(const tensorflow::Tensor &tensor, const tensorflow::TensorProto &proto) const;

private:
    void registerTensorMemoryLocked(tensorflow::Tensor *tensor);

    tensorflow::SessionOptions m_options;

    TFOpLibrary *m_oplibrary;

    std::string m_sessHandle;

    tensorflow::OpSegment m_opseg;
    std::vector<std::unique_ptr<tensorflow::OpKernel>> m_kernels;

    tensorflow::FunctionLibraryDefinition m_flibDef;
    std::unique_ptr<tensorflow::FunctionLibraryRuntime> m_fruntime;
    std::unique_ptr<TFDevice> m_device;

    tensorflow::mutex m_mu;
    std::unordered_map<uint64_t, tensorflow::Tensor*> m_tensors;
};

#endif // TFSESSION_H
