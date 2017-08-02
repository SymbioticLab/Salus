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

#include "execution/devices.h"

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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <list>

using tensorflow::TensorValue;
typedef tensorflow::gtl::InlinedVector<tensorflow::TensorValue, 4> TensorValueVec;
typedef tensorflow::gtl::InlinedVector<tensorflow::DeviceContext*, 4> DeviceContextVec;
typedef tensorflow::gtl::InlinedVector<tensorflow::AllocatorAttributes, 4> AllocatorAttributeVec;

namespace tensorflow {
class OptimizerOptions;
class NodeDef;
class FunctionDefLibrary;
class ConfigProto;
class Graph;
class Node;
namespace remote {
class GraphView;
struct NodeItem;
} // namespace remote
} // namespace tensorflow

class TFRendezvous;
class TFAllocator;
class TFDevice;
class TFSession;
class TFExecutionState;
class TFOpLibrary;

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

class TFContext
{
    std::unique_ptr<tensorflow::OpKernelContext> context;

public:
    TFContext(TFExecutionState *exec, uint64_t taskId);
    ~TFContext();

    tensorflow::OpKernelContext *ctx();

    TFExecutionState *execState;

    uint64_t seq;

    tensorflow::ScopedStepContainer step_container;
    tensorflow::checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    std::unique_ptr<TFRendezvous> rendez;

    tensorflow::TensorStore tensorStore;

    // Holds tensors that are created due to automatic deref
    // Must use a list because we take address of these elements
    // so they must remain valid after later insertion.
    std::list<tensorflow::Tensor> deref_tensors;

    // Which device this context runs on
    DeviceSpec devSpec;

    TensorValueVec inputs;
    DeviceContextVec input_device_contexts;
    AllocatorAttributeVec input_alloc_attrs;

    // A reference to NodeItem in GraphView
    tensorflow::remote::NodeItem *node_item;

    tensorflow::OpKernelContext::Params params;
};

class TFExecutionState
{
public:
    explicit TFExecutionState(TFSession *sess, const std::string &execId, tensorflow::GraphDef &&graphdef,
                              const tensorflow::OptimizerOptions &optOptions);
    bool initialize();

    ~TFExecutionState();

    const std::string &execId() const;

    tensorflow::FunctionLibraryRuntime *functionRuntime(tensorflow::Device *tfdev);

    tensorflow::DeviceContext *deviceContext(const std::string &name, tensorflow::Device *tfdev);

    TFSession *session() const;

    tensorflow::Rendezvous *rendez() const;

    tensorflow::Graph *graph() const;
    tensorflow::remote::GraphView *gview() const;
    tensorflow::Node *findNodeInGraph(const std::string &name) const;
    tensorflow::remote::NodeItem *prepareNodeItemOnDevice(tensorflow::OpKernel *opkernel,
                                                          tensorflow::Device *d);

private:
    TFSession *m_session;
    std::string m_execId;

    tensorflow::GraphDef m_graphdef;

    std::unique_ptr<tensorflow::FunctionLibraryDefinition> m_fdefinition;
    std::unique_ptr<tensorflow::Graph> m_graph;
    std::unique_ptr<tensorflow::remote::GraphView> m_gview;
    std::unordered_map<std::string, int> m_gindex;

    tensorflow::OptimizerOptions m_optOptions;

    tensorflow::Rendezvous *m_rendez;

    tensorflow::mutex m_mu;
    std::unordered_map<tensorflow::Device*, std::unique_ptr<tensorflow::FunctionLibraryRuntime>> m_fruntimes;
    std::unordered_map<tensorflow::Device*, tensorflow::DeviceContextMap> m_deviceContexts;
};

class TFSession
{
public:
    TFSession(TFOpLibrary *opLibrary, const std::string &sessionId,
              const tensorflow::ConfigProto &configProto);

    bool initialize();

    ~TFSession();

    TFExecutionState *prepareExecution(tensorflow::GraphDef &&graphdef);
    TFExecutionState *findExecution(const std::string &execId);

    /**
     * Find or create a kernel on dev.
     * If a matching kernel found or created successfully, returns true, with kernel set to not null
     * If a kernel is found but on nonmatching device, returns false, kernel set to not null, dev set to
     *   previous device
     * Otherwise, returns false with kernel set up nullptr
     */
    bool findOrCreateKernel(TFExecutionState *execState, const tensorflow::NodeDef &ndef,
                            tensorflow::OpKernel *&kernel, DeviceSpec &dev);

    std::unique_ptr<TFContext> createContext(const executor::TFOpContextDef &tfdef,
                                             tensorflow::OpKernel *opkernel, uint64_t taskId,
                                             TFExecutionState *execState, const DeviceSpec &dev);
    void registerContext(uint64_t taskId, TFContext *ctx);
    TFContext *findContext(uint64_t taskId);
    void deregisterContext(uint64_t taskId);

    // Tensor memory management
    struct TensorItem
    {
        TensorValue val;
        tensorflow::DeviceContext *context;
        tensorflow::Device *device;
        tensorflow::remote::NodeItem *nodeItem;
        int slot; // which output is this tensor item in nodeItem

        inline tensorflow::AllocatorAttributes allocAttr() const;
    };
    bool findTensorFromName(const std::string &name, TensorItem &val);
    void registerTensorForName(const std::string &name, TensorItem val);

    void tensorToProtoMeta(tensorflow::TensorProto *meta, TensorValue val);
    void tensorToProtoData(tensorflow::TensorProto *data, TensorValue val);

    bool isCompatible(const tensorflow::Tensor &tensor, const tensorflow::TensorProto &proto) const;

    // Helper methods for run
    executor::TFOpContextUpdate finalizeContext(TFContext *pContext);

    // Accessors
    const tensorflow::SessionOptions &sessionOptions() const { return m_options; }
    tensorflow::Device *findDevice(const DeviceSpec &dev) const;

private:
    TFOpLibrary *m_oplibrary;

    std::string m_sessionId;

    tensorflow::SessionOptions m_options;

    std::unique_ptr<TFAllocator> m_cpuAllocator;

    tensorflow::OpSegment m_opseg;
    /**
     * Remember the device for cached stateful kernels
     */
    tensorflow::mutex m_muk;
    std::unordered_map<tensorflow::OpKernel*, DeviceSpec> m_kernelDevice;
    std::vector<std::unique_ptr<tensorflow::OpKernel>> m_kernels;

    tensorflow::SessionState m_sessState;

    tensorflow::mutex m_mu;
    std::unordered_map<std::string, TensorItem> m_tensors;

    tensorflow::mutex m_muctx;
    /**
     * Mapping RunRequest seq number to TFContext. Protected by m_muctx.
     */
    std::unordered_map<uint64_t, TFContext*> m_contexts;

    tensorflow::mutex m_muexec;
    std::unordered_map<std::string, std::unique_ptr<TFExecutionState>> m_execStates;
};

#endif // TFSESSION_H
