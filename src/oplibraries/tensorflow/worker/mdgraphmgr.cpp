/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Peifeng Yu <peifeng@umich.edu>
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
 */

#include "mdgraphmgr.h"
#include "oplibraries/tensorflow/v2/md_executor.h"
#include "utils/pointerutils.h"

namespace tf = ::tensorflow;

namespace salus::oplib::tensorflow {

MDGraphMgr::MDGraphMgr(const tf::WorkerEnv *env, tf::DeviceMgr *device_mgr, ExecutionContext execCtx)
    : GraphMgr(env, device_mgr)
    , m_execCtx(std::move(execCtx))
{
}

MDGraphMgr::~MDGraphMgr() = default;

Status MDGraphMgr::InitItem(const std::string &session, const tf::GraphDef &gdef,
                            const tf::GraphOptions &graph_options, const tf::DebugOptions &debug_options,
                            tf::DistributedFunctionLibraryRuntime *cluster_flr, Item *item)
{
    item->session = session;
    item->lib_def = std::make_unique<tf::FunctionLibraryDefinition>(tf::OpRegistry::Global(), gdef.library());

    if (gdef.versions().producer() >= 5) {
        // Validate the graph: we assume that merging two valid graphs
        // should maintain graph validity.
        TF_RETURN_IF_ERROR(tf::graph::ValidateGraphDef(gdef, *item->lib_def));
    }

    item->proc_flr =
        std::make_unique<tf::ProcessFunctionLibraryRuntime>(device_mgr_, worker_env_->env,
                                                            gdef.versions().producer(), item->lib_def.get(),
                                                            graph_options.optimizer_options(), cluster_flr);

    // Constructs the graph out of "gdef".
    tf::Graph graph(tf::OpRegistry::Global());
    tf::GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, gdef, &graph));

    // Splits "graph" into two subgraphs, one on client side, containing only inputs,
    // the other on salus, containing everything else.
    std::unordered_map<std::string, tf::GraphDef> partitions;
    tf::PartitionOptions popts;
    // FIXME: verify this works by looking into the Partition function
    popts.node_to_loc = [](auto node) {
        if (node->assigned_device_name() == "cpu:0") {
            return "cpu:0";
        }
        return "salus";
    };
    popts.new_name = [this](const auto &prefix) {
        tf::mutex_lock l(mu_);
        return tf::strings::StrCat(prefix, "_G", next_id_++);
    };
    popts.get_incarnation = [deviceMgr = device_mgr_](const auto &name) -> uint64_t
    {
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
    for (const auto & [key, partdef] : partitions) {
        auto device_graph = std::make_unique<tf::Graph>(tf::OpRegistry::Global());
        tf::GraphConstructorOptions device_opts;
        // There are internal operations (e.g., send/recv) that we now allow.
        device_opts.allow_internal_ops = true;
        device_opts.expect_device_spec = true;
        TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(device_opts, partdef, device_graph.get()));
        partition_graphs.emplace(key, std::move(device_graph));
    }

    // Run optimization on each partition
    tf::GraphOptimizationPassOptions optimization_options;
    optimization_options.flib_def = item->lib_def.get();
    optimization_options.partition_graphs = &partition_graphs;
    TF_RETURN_IF_ERROR(
        tf::OptimizationPassRegistry::Global()->RunGrouping(tf::OptimizationPassRegistry::POST_PARTITIONING,
                                                            optimization_options));

    MultiDeviceExecutorParams params(*worker_env_->device_mgr, m_resourceMgr);
    params.session = session;

    item->units.reserve(partitions.size());
    item->graph_mgr = this;
    const auto &optimizer_opts = graph_options.optimizer_options();
    // FIXME: see if anything is done in graph optimizer
    // FIXME: see if MaybeRewriteGraph has done anything
    tf::GraphOptimizer optimizer(optimizer_opts);
    for (auto & [key, subgraph] : partition_graphs) {
        if (key == "salus") {
            continue;
        }

        auto &unit = item->units.emplace_back();

        // Find the device.
        auto s = device_mgr_->LookupDevice(key, &unit.device);
        if (!s.ok()) {
            // Remove the empty unit from the item as the item destructor wants all
            // units to have valid devices.
            item->units.pop_back();
            return s;
        }

        // Give the device an opportunity to rewrite its subgraph.
        TF_RETURN_IF_ERROR(unit.device->MaybeRewriteGraph(&subgraph));

        // Top-level nodes in the graph uses the op segment to cache
        // kernels. Therefore, as long as the executor is alive, we need
        // to ensure the kernels cached for the session are alive.
        // FIXME: why use global shared op_segment?
        // auto opseg = unit->device->op_segment();
        auto &opseg = m_opseg;
        opseg.AddHold(session);

        auto producer = subgraph->versions().producer();
        params.create_fruntime = [worker_env = worker_env_, producer, item, optimizer_opts](auto dev)
        {
            item->Ref();
            auto flib = tf::NewFunctionLibraryRuntime(worker_env->device_mgr, worker_env->env, dev, producer, item->lib_def.get(), optimizer_opts, item->proc_flr.get());
            return std::shared_ptr<tf::FunctionLibraryRuntime>(flib.release(), [a = sstl::wrap_unref(item)](auto r) { delete r; });
        };

        // Construct the root executor for the subgraph.
        params.find_kernel = [this, session, &opseg](const auto &ndef, auto *devName, auto **kernel) {
            *kernel = nullptr;
            devName->clear();

            bool found = true;
            auto ok = opseg.FindOrCreate(session, ndef.name(), kernel, [&found](auto) {
                found = false;
                return tf::Status::OK();
            });
            if (!ok.ok() || !found) {
                return ok;
            }

            sstl::Guard l(m_mu);
            auto it = m_kernelToDevice.find(*kernel);
            if (it == m_kernelToDevice.end()) {
                return tf::errors::Internal("We've created the kernel, but don't remember its device");
            }
            *devName = it->second;
            return tf::Status::OK();
        };

        params.create_kernel = [this, session, &opseg](const auto &ndef, auto *lib, auto **kernel) {
            // We do not share the kernel via the OpSegment if the node is
            // stateless, or a function.
            // NOTE(mrry): We must not share function kernels (implemented
            // using `CallOp`) between subgraphs, because `CallOp::handle_`
            // is tied to a particular subgraph. Even if the function itself
            // is stateful, the `CallOp` that invokes it is not.
            if (!lib->IsStateful(ndef.op())
                || lib->GetFunctionLibraryDefinition()->Find(ndef.op()) != nullptr) {
                return lib->CreateKernel(ndef, kernel);
            }

            auto create_fn = [this, lib, &ndef](auto **kernel) {
                auto s = lib->CreateKernel(ndef, kernel);
                sstl::Guard l(m_mu);
                m_kernelToDevice[*kernel] = lib->device()->name();
                return s;
            };
            // Kernels created for subgraph nodes need to be cached.  On
            // cache miss, create_fn() is invoked to create a kernel based
            // on the function library here + global op registry.
            return opseg.FindOrCreate(session, ndef.name(), kernel, create_fn);
        };

        params.delete_kernel = [](auto *kernel, auto *lib) {
            // If the node is stateful, opseg owns it. Otherwise, delete it.
            if (kernel && !lib->IsStateful(kernel->type_string())) {
                delete kernel;
            }
        };

        unit.lib = item->proc_flr->GetFLR(unit.device->name());

        optimizer.Optimize(unit.lib, worker_env_->env, unit.device, &subgraph, /*shape_map=*/nullptr);

        // EXPERIMENTAL: tfdbg inserts debug nodes (i.e., probes) to the graph.
        if (!debug_options.debug_tensor_watch_opts().empty()) {
            TF_RETURN_IF_ERROR(DecorateAndPublishGraphForDebug(debug_options, subgraph.get(), unit.device));
        }

        TF_RETURN_IF_ERROR(
            tf::EnsureMemoryTypes(tf::DeviceType(unit.device->device_type()), unit.device->name(), subgraph.get()));
        unit.graph = subgraph.get();
        unit.build_cost_model = graph_options.build_cost_model();

        // NOTE: Always skip cost models, which causes a deadlock
        // when calling item->Unref() from delete_fruntime.
        skip_cost_models_ = true;

        params.ins = m_execCtx;
        TF_RETURN_IF_ERROR(NewMultiDeviceExecutor(params, subgraph.release(), &unit.root));
    }
    return tf::Status::OK();
}

} // namespace salus::oplib::tensorflow
