//
// Created by peifeng on 7/24/18.
//

#ifndef EXECUTOR_TF_EXECUTOR_H
#define EXECUTOR_TF_EXECUTOR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "execution/executionengine.h"

namespace salus::oplib::tensorflow {

using POpKernel = std::unique_ptr<tf::OpKernel, void (*)(tf::OpKernel *)>;

struct TFExecutorParams
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
//    std::function<Status(const tf::NodeDef &, POpKernel *)> create_kernel;
    std::function<Status(const tf::NodeDef&, tf::OpKernel**)> create_kernel;
    std::function<void(tf::OpKernel*)> delete_kernel;

    tf::Executor::Args::NodeOutputsCallback node_outputs_cb;
};

Status NewTFExecutor(TFExecutorParams params, std::unique_ptr<const tf::Graph> &&graph,
                     tf::Executor **executor);
Status CreateNonCachedKernel(tf::Device* device, tf::FunctionLibraryRuntime* flib,
                             const tf::NodeDef& ndef, int graph_def_version,
                             tf::OpKernel** kernel);

// Deletes "kernel" returned by CreateKernel.
void DeleteNonCachedKernel(tf::OpKernel* kernel);

} // namespace salus::oplib::tensorflow


#endif //EXECUTOR_TF_EXECUTOR_H
