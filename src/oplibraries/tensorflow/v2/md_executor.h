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
tensorflow::Status NewMultiDeviceExecutor(const tf::MultiDeviceExecutorParams& params,
                                          const tf::Graph* graph, ExecutionEngine::Inserter ins,
                                          tf::Executor **executor);

#endif  // OPLIBRARIES_TENSORFLOW_MULTI_DEVICE_EXECUTOR_H_
