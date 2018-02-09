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

/*
 * Make sure tensorflow_headers is included first before
 * any other headers, so we can correctly override TF logging
 * with ours.
 */
#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "execution/executionengine.h"
#include "utils/pointerutils.h"
#include <functional>

namespace tensorflow {
class OpSegment;
class NodeDef;
class OpKernel;
class Device;
class DeviceMgr;
} // namespace tensorflow

namespace symbiotic::salus::oplib::tensorflow {
struct MultiDeviceExecutorParams
{
    std::string session;

    // The devices this executor should use.
    not_null<tf::DeviceMgr> deviceMgr;

    // The resource manager this executor should use.
    not_null<tf::ResourceMgr> resourceMgr;

    // create_fruntime creates function library runtime given device,
    // caller takes the ownership of the created library runtime.
    std::function<tf::FunctionLibraryRuntime *(tf::Device *)> create_fruntime;
    std::function<void(tf::FunctionLibraryRuntime *)> delete_fruntime;

    // find_kernel returns an instance of op kernel, which was created on device.
    // create_kernel returns an instance of op kernel based on NodeDef for device d.
    // delete_kernel is called for every kernel used by the executor
    // when the executor is deleted.
    std::function<Status(const tf::NodeDef &, std::string *, tf::OpKernel **)> find_kernel;

    std::function<Status(const tf::NodeDef &, tf::FunctionLibraryRuntime *, tf::OpKernel **)> create_kernel;

    std::function<void(tf::OpKernel *, tf::FunctionLibraryRuntime *)> delete_kernel;

    tf::Executor::Args::NodeOutputsCallback node_outputs_cb;
};

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor".
// The caller keeps the ownership of "device".
// The returned executor takes the ownership of "graph".
// Otherwise, returns an error status.
Status NewMultiDeviceExecutor(const MultiDeviceExecutorParams &params, const tf::Graph *graph,
                              ExecutionContext ins, tf::Executor **executor);

} // namespace symbiotic::salus::oplib::tensorflow

#endif // OPLIBRARIES_TENSORFLOW_MULTI_DEVICE_EXECUTOR_H_
