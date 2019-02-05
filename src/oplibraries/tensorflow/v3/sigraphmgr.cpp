/*
 * Copyright 2019 Peifeng Yu <peifeng@umich.edu>
 * 
 * This file is part of Salus
 * (see https://github.com/SymbioticLab/Salus).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "oplibraries/tensorflow/v3/sigraphmgr.h"

#include "oplibraries/tensorflow/device/salusdevices.h"
#include "oplibraries/tensorflow/device/shadowdevices.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "oplibraries/tensorflow/v3/tf_executor.h"
#include "oplibraries/tensorflow/worker/dummyworkercache.h"

namespace salus::oplib::tensorflow {

SIGraphMgr::SIGraphMgr(const tf::WorkerEnv *env, tf::DeviceMgr *mgr, std::shared_ptr<ExecutionContext> execCtx)
    : GraphMgr(env, mgr)
    , m_execCtx(std::move(execCtx))
{
}

SIGraphMgr::~SIGraphMgr()
{
    // We have to do this before m_resourceMgr goes out of scope
    for (auto p : table_) {
        p.second->Unref();
    }
    table_.clear();
}

Status SIGraphMgr::Register(const std::string &session, const tf::GraphDef &gdef, const tf::GraphOptions &graph_options,
                            const tf::DebugOptions &debug_options, tf::DistributedFunctionLibraryRuntime *cluster_flr,
                            std::string *handle)
{
    auto item = sstl::make_scoped_unref<Item>();
    std::string tempHandle;
    {
        tf::mutex_lock l(mu_);
        tempHandle = tf::strings::Printf("%016llx", ++next_id_);
        item->handle = tempHandle;
    }

    TF_RETURN_IF_ERROR(InitSIItem(session, gdef, graph_options, debug_options, cluster_flr, *item));

    *handle = std::move(tempHandle);
    // Inserts one item into table_
    {
        tf::mutex_lock l(mu_);
        // This release transfers one ref into table_
        CHECK(table_.insert({*handle, item.release()}).second);
    }
    return Status::OK();
}

tf::Status SIGraphMgr::InitSIItem(const std::string &session, const tf::GraphDef &gdef,
                                  const tf::GraphOptions &graph_options, const tf::DebugOptions &debug_options,
                                  tf::DistributedFunctionLibraryRuntime *cluster_flr, Item &item)
{
    item.session = session;
    item.lib_def = std::make_unique<tf::FunctionLibraryDefinition>(tf::OpRegistry::Global(), gdef.library());

    if (gdef.versions().producer() >= 5) {
        // Validate the graph: we assume that merging two valid graphs
        // should maintain graph validity.
        TF_RETURN_IF_ERROR(tf::graph::ValidateGraphDef(gdef, *item.lib_def));
    }

    item.proc_flr = std::make_unique<tf::ProcessFunctionLibraryRuntime>(device_mgr_, worker_env_->env,
                                                                        gdef.versions().producer(), item.lib_def.get(),
                                                                        graph_options.optimizer_options(), cluster_flr);

    // Constructs the graph out of "gdef"
    tf::Graph graph(tf::OpRegistry::Global());
    tf::GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, gdef, &graph));

    // Make sure salus_main_iter marker is on GPU
    for (auto *n : graph.nodes()) {
        if (n->name() == "salus_main_iter") {
            n->set_assigned_device_name("/job:salus/replica:0/task:0/device:GPU:0");
        }
    }

    // FUTURE: split into only two subgraphs, one on client side, containing only inputs,
    // the other on salus, containing everything else.

    // Splits "graph" into multiple subgraphs by device names.
    std::unordered_map<std::string, tf::GraphDef> partitions;
    tf::PartitionOptions popts;
    popts.node_to_loc = [](auto node) { return node->assigned_device_name(); };
    popts.new_name = [this](const auto &prefix) {
        tf::mutex_lock l(mu_);
        return tf::strings::StrCat(prefix, "_G", next_id_++);
    };
    popts.get_incarnation = [deviceMgr = device_mgr_](const auto &name) -> uint64_t {
        tf::Device *device = nullptr;
        auto s = deviceMgr->LookupDevice(name, &device);
        if (s.ok()) {
            return device->attributes().incarnation();
        } else {
            return tf::PartitionOptions::kIllegalIncarnation;
        }
    };
    popts.flib_def = &graph.flib_def();
    popts.control_flow_added = true;
    popts.scheduling_for_recvs = graph_options.enable_recv_scheduling();
    TF_RETURN_IF_ERROR(Partition(popts, &graph, &partitions));
    if (popts.scheduling_for_recvs) {
        TF_RETURN_IF_ERROR(AddControlEdges(popts, &partitions));
    }

    std::unordered_map<std::string, std::unique_ptr<tf::Graph>> partition_graphs;
    for (const auto &[key, partdef] : partitions) {
        auto device_graph = std::make_unique<tf::Graph>(tf::OpRegistry::Global());
        tf::GraphConstructorOptions device_opts;
        // There are internal operations (e.g. send/recv) that we now allow.
        device_opts.allow_internal_ops = true;
        device_opts.expect_device_spec = true;
        TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(device_opts, partdef, device_graph.get()));
        partition_graphs.emplace(key, std::move(device_graph));
    }

    tf::GraphOptimizationPassOptions optimization_options;
    optimization_options.flib_def = item.lib_def.get();
    optimization_options.partition_graphs = &partition_graphs;
    TF_RETURN_IF_ERROR(
        tf::OptimizationPassRegistry::Global()->RunGrouping(tf::OptimizationPassRegistry::POST_PARTITIONING,
                                                            optimization_options));

    TFExecutorParams params;
    params.session = session;
    params.graphHandle = item.handle;

    item.units.reserve(partitions.size());
    item.graph_mgr = this;
    const auto &optimizer_opts = graph_options.optimizer_options();
    tf::GraphOptimizer optimizer(optimizer_opts);
    for (auto &[key, subgraph] : partition_graphs) {
        auto &unit = item.units.emplace_back();

        // Find the device
        auto s = device_mgr_->LookupDevice(key, &unit.device);
        if (!s.ok()) {
            // Remove the empty unit from the item as the item destructor wants all
            // units to have valid devices.
            item.units.pop_back();
            return s;
        }

        // Give the device an opportunity to rewrite its subgraph.
        TF_RETURN_IF_ERROR(unit.device->MaybeRewriteGraph(&subgraph));

        // Top-level nodes in the graph uses the op segment to cache
        // kernels. Therefore, as long as the executor is alive, we need
        // to ensure the kernels cached for the session are alive.
        auto opseg = unit.device->op_segment();
        opseg->AddHold(session);

        // Function library runtime.
        auto lib = item.proc_flr->GetFLR(unit.device->name());
        if (!lib) {
            return tf::errors::InvalidArgument("Cannot find FLR for device: ", unit.device->name());
        }

        // Construct the root executor for the subgraph
        params.device = unit.device;
        params.function_library = lib;
        params.create_kernel = [session, lib, opseg](const auto &ndef, tf::OpKernel **kernel) {
            // We do not share the kernel via the OpSegment if the node is
            // stateless, or a function.
            // NOTE(mrry): We must not share function kernels (implemented
            // using `CallOp`) between subgraphs, because `CallOp::handle_`
            // is tied to a particular subgraph. Even if the function itself
            // is stateful, the `CallOp` that invokes it is not.
            if (!lib->IsStateful(ndef.op()) || lib->GetFunctionLibraryDefinition()->Find(ndef.op()) != nullptr) {
                return lib->CreateKernel(ndef, kernel);
            }
            auto create_fn = [lib, &ndef](tf::OpKernel **kernel) { return lib->CreateKernel(ndef, kernel); };
            // Kernels created for subgraph nodes need to be cached.  On
            // cache miss, create_fn() is invoked to create a kernel based
            // on the function library here + global op registry.
            return opseg->FindOrCreate(session, ndef.name(), kernel, create_fn);
        };
        params.delete_kernel = [lib](tf::OpKernel *kernel) {
            // If the node is stateful, opseg owns it. Otherwise, delete it.
            if (kernel && !lib->IsStateful(kernel->type_string())) {
                delete kernel;
            }
        };

        optimizer.Optimize(lib, worker_env_->env, params.device, &subgraph, /*shape_map=*/nullptr);

        // EXPERIMENTAL: tfdbg inserts debug nodes (i.e., probes) to the graph.
        if (!debug_options.debug_tensor_watch_opts().empty()) {
            TF_RETURN_IF_ERROR(DecorateAndPublishGraphForDebug(debug_options, subgraph.get(), params.device));
        }

        TF_RETURN_IF_ERROR(
            tf::EnsureMemoryTypes(tf::DeviceType(unit.device->device_type()), unit.device->name(), subgraph.get()));
        unit.graph = subgraph.get();
        unit.build_cost_model = graph_options.build_cost_model();
        if (unit.build_cost_model > 0) {
            skip_cost_models_ = false;
        }

        params.ins = m_execCtx;
        TF_RETURN_IF_ERROR(NewTFExecutor(params, std::move(subgraph), &unit.root));
    }
    return Status::OK();
}

CreateWorkerSessionFn GetCreateWorkerSessionFnForSIGraphMgr(const std::string &worker_name,
                                                            const tf::WorkerEnv *worker_env,
                                                            std::shared_ptr<ExecutionContext> execCtx,
                                                            const tf::ConfigProto &)
{
    return [=, ctx = std::move(execCtx)](const auto &sessHandle) {
        // Create shadow devices for isolated sessions, deviceMgr owns the passed in devices
        std::vector<tf::Device *> renamedDevices;
        for (auto d : worker_env->local_devices) {
            auto sd = ISalusDevice::safe_cast(d);
            renamedDevices.push_back(sd->createSessionDevice(worker_name, sessHandle).release());
        }
        auto deviceMgr = std::make_unique<tf::DeviceMgr>(renamedDevices);
        auto graphMgr = std::make_unique<SIGraphMgr>(worker_env, deviceMgr.get(), ctx);
        return std::make_unique<tf::WorkerSession>(sessHandle, worker_name,
                                                   // This is never used, but internally WorkerSession creates
                                                   // a wrapper around it. So we have to supply an empty one
                                                   std::make_unique<EmptyWorkerCache>(), std::move(deviceMgr),
                                                   std::move(graphMgr));
    };
}
} // namespace salus::oplib::tensorflow
