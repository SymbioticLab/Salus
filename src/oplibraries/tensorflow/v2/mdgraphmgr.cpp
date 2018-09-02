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

#include "oplibraries/tensorflow/v2/mdgraphmgr.h"

#include "oplibraries/tensorflow/device/salusdevices.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/v2/md_executor.h"
#include "oplibraries/tensorflow/worker/dummyworkercache.h"
#include "utils/pointerutils.h"

namespace salus::oplib::tensorflow {

constexpr void skip_delete_opkernel(tf::OpKernel *) {}
constexpr void default_delete_opkernel(tf::OpKernel *k)
{
    delete k;
}

MDGraphMgr::MDGraphMgr(const tf::WorkerEnv *env, std::shared_ptr<ExecutionContext> execCtx, ResStats rm)
    : GraphMgr(env, env->device_mgr)
    , m_execCtx(std::move(execCtx))
    , m_rm(rm)
{
}

MDGraphMgr::~MDGraphMgr()
{
    // We have to do this before m_resourceMgr goes out of scope
    for (auto p : table_)
        p.second->Unref();
    table_.clear();
}

MDGraphMgr::MDItem::~MDItem()
{
    // Manually clear units
    // because ExecutorImpl holds some ditem, which in turn holds some flr, which
    // should be deleted before this goes out of scope.
    for (const auto &unit : units) {
        CHECK_NOTNULL(unit.device);
        delete unit.root;
    }
    // Clear the list so Item::~Item won't do anything
    units.clear();

    for (auto dev : devices) {
        dev->op_segment()->RemoveHold(session);
    }
}

Status MDGraphMgr::Register(const std::string &session, const tf::GraphDef &gdef,
                            const tf::GraphOptions &graph_options, const tf::DebugOptions &debug_options,
                            tf::DistributedFunctionLibraryRuntime *cluster_flr, std::string *handle)
{
    auto item = sstl::make_scoped_unref<MDItem>();
    std::string tempHandle;
    {
        tf::mutex_lock l(mu_);
        tempHandle = tf::strings::Printf("%016llx", ++next_id_);
        item->handle = tempHandle;
    }

    auto s = InitMDItem(session, gdef, graph_options, debug_options, cluster_flr, item.get());
    if (!s.ok()) {
        return s;
    }

    *handle = tempHandle;
    // Inserts one item into table_.
    {
        tf::mutex_lock l(mu_);
        // This release transfers one ref into table_
        CHECK(table_.emplace(*handle, item.release()).second);
    }
    return Status::OK();
}

Status MDGraphMgr::InitMDItem(const std::string &session, const tf::GraphDef &gdef,
                              const tf::GraphOptions &graph_options, const tf::DebugOptions &debug_options,
                              tf::DistributedFunctionLibraryRuntime *cluster_flr, MDItem *item)
{
    item->session = session;
    item->devices = device_mgr_->ListDevices();

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

    // Make sure salus_main_iter marker is on GPU
    for (auto *n : graph.nodes()) {
        if (n->name() == "salus_main_iter") {
            n->set_assigned_device_name("/job:salus/replica:0/task:0/device:GPU:0");
        }
    }

    // Splits "graph" into two subgraphs, one on client side, containing only inputs,
    // the other on salus, containing everything else.
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
    params.graphHandle = item->handle;

    item->units.reserve(partitions.size());
    item->graph_mgr = this;
    const auto &optimizer_opts = graph_options.optimizer_options();
    tf::GraphOptimizer optimizer(optimizer_opts);
    for (auto &[key, subgraph] : partition_graphs) {
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

        // Add a hold on op_segment on every device
        // These holds are removed in ~MDItem.
        for (auto tfdev : item->devices) {
            tfdev->op_segment()->AddHold(session);
        }

        auto producer = subgraph->versions().producer();
        params.create_fruntime = [worker_env = worker_env_, producer, item, optimizer_opts](auto dev) {
            auto flib =
                tf::NewFunctionLibraryRuntime(worker_env->device_mgr, worker_env->env, dev, producer,
                                              item->lib_def.get(), optimizer_opts, item->proc_flr.get());
            return std::shared_ptr<tf::FunctionLibraryRuntime>(std::move(flib));
        };

        params.get_kernel = [item](const auto &ndef, auto *lib) -> POpKernel {
            tf::OpKernel *kernel = nullptr;
            // We do not share the kernel via the OpSegment if the node is
            // stateless, or a function.
            // NOTE(mrry): We must not share function kernels (implemented
            // using `CallOp`) between subgraphs, because `CallOp::handle_`
            // is tied to a particular subgraph. Even if the function itself
            // is stateful, the `CallOp` that invokes it is not.
            if (!lib->IsStateful(ndef.op())
                || lib->GetFunctionLibraryDefinition()->Find(ndef.op()) != nullptr) {
                SALUS_THROW_IF_ERROR(lib->CreateKernel(ndef, &kernel));
                VLOG(2) << "Using noncached kernel " << ndef.name() << "@" << as_hex(kernel) << " on device "
                        << lib->device()->name();
                return {kernel, default_delete_opkernel};
            }

            // the kernel should hold a reference for item, but the function library runtime
            // already holds one, so to simplify things, we omit the item->Ref() and Unref
            // for kernel
            auto create_fn = [item, lib, &ndef](auto **pkernel) {
                auto ok = lib->CreateKernel(ndef, pkernel);
                VLOG(2) << "Creating cached kernel " << ndef << "@" << as_hex(*pkernel) << " on device "
                        << lib->device()->name() << " for graphHandle=" << item->handle;
                return ok;
            };

            // Cache the kernel in underlaying device's op segment, which has separate storage per session
            auto &tfdev = static_cast<PerTaskDevice *>(lib->device())->underlayingDevice();
            // On cache miss, create_fn() is invoked to create a kernel based
            // on the function library here + global op registry.
            SALUS_THROW_IF_ERROR(
                tfdev.op_segment()->FindOrCreate(item->session, ndef.name(), &kernel, create_fn));
            VLOG(2) << "Using cached kernel " << ndef << "@" << as_hex(kernel) << " on device "
                    << tfdev.name() << " for graphHandle=" << item->handle;
            return {kernel, skip_delete_opkernel};
        };

        unit.lib = item->proc_flr->GetFLR(unit.device->name());

        optimizer.Optimize(unit.lib, worker_env_->env, unit.device, &subgraph, /*shape_map=*/nullptr);

        // EXPERIMENTAL: tfdbg inserts debug nodes (i.e., probes) to the graph.
        if (!debug_options.debug_tensor_watch_opts().empty()) {
            TF_RETURN_IF_ERROR(DecorateAndPublishGraphForDebug(debug_options, subgraph.get(), unit.device));
        }

        TF_RETURN_IF_ERROR(tf::EnsureMemoryTypes(tf::DeviceType(unit.device->device_type()),
                                                 unit.device->name(), subgraph.get()));
        unit.graph = subgraph.get();
        unit.build_cost_model = graph_options.build_cost_model();

        // NOTE: Always skip cost models, which causes a deadlock
        // when calling item->Unref() from delete_fruntime.
        skip_cost_models_ = true;

        params.ins = m_execCtx;
        params.rm = m_rm;
        TF_RETURN_IF_ERROR(NewMultiDeviceExecutor(params, std::move(subgraph), &unit.root));
    }
    return tf::Status::OK();
}

CreateWorkerSessionFn GetCreateWorkerSessionFnForMDGraphMgr(const std::string &worker_name,
                                                            const tf::WorkerEnv *worker_env,
                                                            std::shared_ptr<ExecutionContext> execCtx,
                                                            const tf::ConfigProto &config)
{
    // Get resource estimation from client
    ResStats rm{};
    const auto rt = "MEMORY:GPU";
    auto &m = config.salus_options().resource_map();
    if (auto oval = sstl::optionalGet(m.persistant(), rt)) {
        rm.persist = static_cast<size_t>(std::round(*oval));
    }
    if (auto oval = sstl::optionalGet(m.temporary(), rt)) {
        rm.temporary = static_cast<size_t>(std::round(*oval));
    }
    rm.count = 0;

    return [=, ctx = std::move(execCtx)](const auto &sessHandle) {
        // worker session takes ownership of a deviceMgr, so we create shadow devices for it.
        return std::make_unique<tf::WorkerSession>(
            sessHandle, worker_name,
            // This is never used, but internally WorkerSession creates a wrapper
            // around it. So we have to supply an empty one
            std::make_unique<EmptyWorkerCache>(),
            // use an empty device mgr for session, because we should use
            // the device mgr from worker_env to make sure we use ISalusDevice
            std::make_unique<tf::DeviceMgr>(std::vector<tf::Device *>{}),
            std::make_unique<MDGraphMgr>(worker_env, ctx, rm));
    };
}

} // namespace salus::oplib::tensorflow
