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
#include "execution/executionengine.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "utils/pointerutils.h"
#include <functional>
#include <memory>

namespace tensorflow {
class OpSegment;
class NodeDef;
class OpKernel;
class DeviceMgr;
} // namespace tensorflow

namespace salus::oplib::tensorflow {
class PerTaskDevice;

using POpKernel = std::unique_ptr<tf::OpKernel, void (*)(tf::OpKernel *)>;

constexpr void skip_delete_opkernel(tf::OpKernel *) {}
constexpr void default_delete_opkernel(tf::OpKernel *k)
{
    delete k;
}

struct MultiDeviceExecutorParams
{
    MultiDeviceExecutorParams(tf::DeviceMgr &deviceMgr, tf::ResourceMgr &resourceMgr)
        : deviceMgr(deviceMgr)
        , resourceMgr(resourceMgr)
    {
    }

    std::string session;

    ExecutionContext ins;

    // The devices this executor should use.
    tf::DeviceMgr &deviceMgr;

    // The resource manager this executor should use.
    tf::ResourceMgr &resourceMgr;

    // create_fruntime creates function library runtime given device,
    std::function<std::shared_ptr<tf::FunctionLibraryRuntime>(PerTaskDevice *)> create_fruntime;

    /**
     * @brief Get a kernel for nodedef, throws TFException on error
     */
    std::function<POpKernel(const tf::NodeDef &, tf::FunctionLibraryRuntime *)> get_kernel;

    tf::Executor::Args::NodeOutputsCallback node_outputs_cb;
};

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor".
// The returned executor takes the ownership of "graph".
// Otherwise, returns an error status.
Status NewMultiDeviceExecutor(MultiDeviceExecutorParams params, std::unique_ptr<const tf::Graph> &&graph,
                              tf::Executor **executor);

} // namespace salus::oplib::tensorflow

#endif // OPLIBRARIES_TENSORFLOW_MULTI_DEVICE_EXECUTOR_H_
