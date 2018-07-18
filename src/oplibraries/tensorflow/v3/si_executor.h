//
// Created by peifeng on 7/18/18.
//

#ifndef SALUS_OPLIB_TENSORFLOW_SIMPLE_ITER_EXECUTOR_H
#define SALUS_OPLIB_TENSORFLOW_SIMPLE_ITER_EXECUTOR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "execution/executionengine.h"

#include <functional>
#include <memory>

namespace salus::oplib::tensorflow {

using POpKernel = std::unique_ptr<tf::OpKernel, void (*)(tf::OpKernel *)>;

struct SIExecutorParams
{
    std::string session;

    // Graph handle
    std::string graphHandle;

    std::shared_ptr<ExecutionContext> ins;

    tf::Device *device;

    // The library runtime support.
    tf::FunctionLibraryRuntime *function_library = nullptr;

    // create_kernel returns an instance of op kernel based on NodeDef.
    // kernels are deleted when the executor is deleted
    std::function<Status(const tf::NodeDef &, POpKernel *)> create_kernel;

    tf::Executor::Args::NodeOutputsCallback node_outputs_cb;
};

Status NewSIExecutor(SIExecutorParams params, std::unique_ptr<const tf::Graph> &&graph,
                     tf::Executor **executor);

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_SIMPLE_ITER_EXECUTOR_H
