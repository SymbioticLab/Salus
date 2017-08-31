/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef OPLIBRARIES_TENSORFLOW_MULTI_DEVICE_EXECUTOR_H_
#define OPLIBRARIES_TENSORFLOW_MULTI_DEVICE_EXECUTOR_H_

#include <tensorflow/core/common_runtime/executor.h>

#include <functional>

namespace tensorflow {
class OpSegment;
class NodeDef;
class OpKernel;
class Device;
class DeviceMgr;
}

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor".
// The caller keeps the ownership of "device".
// The returned executor takes the ownership of "graph".
// Otherwise, returns an error status.
//
// "params" provides a set of context for the executor. We expect that
// different context would provide different implementations.
struct MultiDeviceExecutorParams {
    // The global op segment for kernel caching.
    tensorflow::OpSegment *opseg;

    // The devices this executor should use.
    tensorflow::DeviceMgr *deviceMgr;

    // create_fruntime creates function library runtime given device,
    // caller takes the ownership of the created library runtime.
    std::function<tensorflow::FunctionLibraryRuntime*(const tensorflow::Device*)> create_fruntime;

    // find_kernel returns an instance of op kernel, which was created on device.
    // create_kernel returns an instance of op kernel based on NodeDef for device d.
    // delete_kernel is called for every kernel used by the executor
    // when the executor is deleted.
    std::function<tensorflow::Status(const tensorflow::NodeDef&,
                                     const tensorflow::Device**,
                                     const tensorflow::OpKernel**)> find_kernel;

    std::function<tensorflow::Status(const tensorflow::NodeDef&,
                                     const tensorflow::FunctionLibraryRuntime*,
                                     const tensorflow::OpKernel**)> create_kernel;

    std::function<void(tensorflow::OpKernel*)> delete_kernel;

    tensorflow::Executor::Args::NodeOutputsCallback node_outputs_cb;
};

tensorflow::Status NewMultiDeviceExecutor(const MultiDeviceExecutorParams& params,
                                          const tensorflow::Graph* graph,
                                          tensorflow::Executor** executor);

#endif  // OPLIBRARIES_TENSORFLOW_MULTI_DEVICE_EXECUTOR_H_
