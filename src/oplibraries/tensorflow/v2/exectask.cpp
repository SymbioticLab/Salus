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
 */

#include "exectask.h"

#include "execution/devices.h"
#include "platform/logging.h"
#include "utils/threadutils.h"
#include "utils/macros.h"
#include "oplibraries/tensorflow/v2/md_rendezvous.h"
#include "oplibraries/tensorflow/v2/tfallocator.h"
#include "oplibraries/tensorflow/v2/peropallocdevice.h"

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include <sstream>

namespace tf = tensorflow;

ExecTask::ExecTask(ExecutorState *state, utils::semaphore &num_finished_ops,
                   ExecutorState::TaggedNode &node, ExecutorState::TaggedNodeSeq &ready,
                   ExecutorState::TaggedNodeReadyQueue &inline_ready,
                   tf::NodeExecStats *stats, tf::OpKernelContext::Params &params,
                   int64_t &scheduled_usec,
                   TensorValueVec &inputs, DeviceContextVec &input_device_contexts,
                   AllocatorAttributeVec &input_alloc_attrs, bool &completed, tf::Rendezvous *rendez,
                   int maxFailures)
    : deleteKernel(state->impl_->params_.delete_kernel)
    , maxFailures(maxFailures)
    , kernel_is_async(false)
    , has_ref_input(false)
    , tagged_node(node)
    , ready(ready)
    , inline_ready(inline_ready)
    , stats(stats)
    , params(params)
    , scheduled_usec(scheduled_usec)
    , inputs(inputs)
    , input_device_contexts(input_device_contexts)
    , input_alloc_attrs(input_alloc_attrs)
    , completed(completed)
    , rendez(rendez)
    , num_finished_ops(num_finished_ops)
    , m_state(state)
{
    tf::DeviceTypeVector tftypes;
    auto ok = tf::SupportedDeviceTypesForNode({tf::DEVICE_GPU, tf::DEVICE_CPU},
                                              tagged_node.node->def(), &tftypes);
    if (!ok.ok()) {
        WARN("Error while querying supported device for node {}: {}", tagged_node.node->name(), ok);
    }

    supportedTypes.reserve(tftypes.size());
    for (auto tft : tftypes) {
        if (tft == tf::DEVICE_CPU) {
            supportedTypes.push_back(DeviceType::CPU);
        } else if (tft == tf::DEVICE_GPU) {
            supportedTypes.push_back(DeviceType::GPU);
        } else {
            WARN("Unknown tf device type: {}", tft.type());
        }
    }

    // pre compute estimated usage
    for (auto t : supportedTypes) {
        estimatedUsage(t);
    }
}

const std::vector<DeviceType> &ExecTask::supportedDeviceTypes() const
{
    return supportedTypes;
}

bool ExecTask::prepare(const std::shared_ptr<ResourceContext> &ctx)
{
    rctx = ctx;

    auto &dev = rctx->spec;

    auto match = [&dev](auto type) { return type == dev.type; };
    if (std::none_of(supportedTypes.begin(), supportedTypes.end(), match)) {
        return false;
    }

    auto s = m_state->impl_->LookupDevice(dev, &ditem);
    if (!s.ok()) {
        return false;
    }

    ditem.device->setResourceContext(rctx);

    // First check if we already created the kernel on some device
    op_kernel = nullptr;
    std::string devName;
    auto ok = m_state->impl_->params_.find_kernel(tagged_node.node->def(), &devName, &op_kernel);

    if (ok.ok() && op_kernel) {
        // we saw this kernel before, check if the device match
        if (devName.empty()) {
            WARN("We've created the kernel, but don't remember its device: {}", tagged_node.node->name());
            auto s = tf::errors::Internal("We've created the kernel, but don't remember its device");
            op_kernel = nullptr;
            return false;
        }
        if (devName == ditem.device->name()) {
            // We are on the same device, good.
            return true;
        }
        TRACE("Stateful kernel can not be moved: previously created on {}, now requested on {}",
              devName, ditem.device->name());
        op_kernel = nullptr;
        return false;
    } else if (!ok.ok()) {
        ERR("Failed to find kernel with status {} for Node: {}", ok, tagged_node.node->name());
        // it is okay, just continue to create the kernel
    }

    return true;
}

Resources ExecTask::estimatedUsage(const DeviceSpec& dev)
{
    // Short-cut if this task has failed before
    if (failureTimes > 0) {
        const auto &sessHandle = m_state->impl_->params_.session;
        auto rm = SessionResourceTracker::instance().usage(sessHandle);
        if (rm) {
            // Merge together
            resources::merge(rm->temporary, rm->persistant);

            auto f = failureTimes;
            if (f > maxFailures) {
                WARN("Failure time exceeds maximum failures: {} (max {})", f, maxFailures);
                f = maxFailures;
            }
            uint64_t scale = 1 << (maxFailures - f);
            resources::scale(rm->temporary, 1.0 / scale);

            // Update cache
            cachedUsage[dev] = rm->temporary;
        } else {
            ERR("No session usage found for exec task: {} under session {}", tagged_node.node->name(), sessHandle);
            // fallback to normal estimation
        }
    }

    // Fast path from cache
    auto it = cachedUsage.find(dev);
    if (it != cachedUsage.end()) {
        return it->second;
    }

    // Slow path to calculate the usage
    auto &res = cachedUsage[dev];

    const auto *node = tagged_node.node;
    auto ctx = m_state->shapeForNode(node);
    if (!ctx) {
        WARN("Shape information not available for node: {}", node->name());
        return res;
    }

    ExecutorImpl::DeviceItem ditem;
    tf::MemoryTypeVector input_mtypes;
    tf::MemoryTypeVector output_mtypes;
    auto mtypeStatus = m_state->impl_->LookupDevice(dev, &ditem);
    if (mtypeStatus.ok()) {
        mtypeStatus.Update(tf::remote::MemoryTypesForNode(m_state->impl_->graph_->op_registry(),
                                                          tf::DeviceType(ditem.device->device_type()),
                                                          node->def(), &input_mtypes, &output_mtypes));
    }
    if (!mtypeStatus.ok()) {
        WARN("Kernel not found on device {}, resource estimation may be inaccurate.", dev);
    }

    ResourceTag devTag{ResourceType::MEMORY, dev};
    ResourceTag cpuTag{ResourceType::MEMORY, dev};

    for (int i = 0; i != ctx->num_outputs(); ++i) {
        auto shp = ctx->output(i);
        if (!ctx->RankKnown(shp)) {
            WARN("{}-th output of node {} has unknown rank", i, node->name());
            continue;
        }
        TRACE("Shape of {}-th output of node {}:", i, node->name());
        size_t count = 1;
        for (int j = 0; j != ctx->Rank(shp); ++j) {
            auto dim = ctx->Dim(shp, j);
            if (!ctx->ValueKnown(dim)) {
                WARN("    Unknown");
                continue;
            }

            auto val = ctx->Value(dim);
            TRACE("    {}", val);
            count *= val;
        }
        auto dtype = node->output_type(i);
        TRACE("    dtype {}, {} bytes", tf::DataType_Name(dtype), tf::DataTypeSize(dtype));
        double subtotal = count * tf::DataTypeSize(dtype);

        if (mtypeStatus.ok() && output_mtypes[i] == tf::HOST_MEMORY) {
            res[cpuTag] += subtotal;
        } else {
            res[devTag] += subtotal;
        }
    }

    return res;
}

std::string ExecTask::DebugString()
{
    std::ostringstream oss;
    oss << "ExecTask(name=" << tagged_node.node->name()
        << ", session=" << m_state->impl_->params_.session
        << ", failures=" << failureTimes << ")";
    return oss.str();
}

void ExecTask::run(Callbacks cbs)
{
    const auto &gview = m_state->impl_->gview_;
    auto node = tagged_node.node;
    auto input_frame = tagged_node.input_frame;
    int64_t input_iter = tagged_node.input_iter;
    const size_t id = node->id();
    const auto &item = *gview.node(id);

    // clear early
    params.rendezvous = nullptr;

    // Instantiate kernel if not ready done
    if (!op_kernel) {
        auto s = m_state->SetupKernel(tagged_node, ditem, &op_kernel);
        if (!s.ok()) {
            ERR("Error when creating kernel for node {}: {}", node->name(), s);
            m_state->MaybeMarkCompleted(input_frame, input_iter, id);
            afterRun(s, cbs);
            return;
        }
    }

    CHECK(op_kernel);
    kernel_is_async = (op_kernel->AsAsync() != nullptr);

    // Go through inputs to see if there's ref type input
    for (int i = 0; i != item.num_inputs; ++i) {
        if (IsRefType(item.input_type(i))) {
            has_ref_input = true;
            break;
        }
    }

    // Start run
    auto s = gview.SetAllocAttrForNode(node, ditem.device.get(), op_kernel);
    if (!s.ok()) {
        m_state->MaybeMarkCompleted(input_frame, input_iter, id);
        afterRun(s, cbs);
        return;
    }

    params.device = ditem.device.get();

    auto localRendez = new MultiDeviceRendezvous(ditem.device, rendez);
    params.rendezvous = localRendez;
    params.record_tensor_accesses = ditem.device_record_tensor_access;
    params.function_library = ditem.function_library.get();
    // Set the device_context for this node id, if it exists.
    params.op_device_context = m_state->FindDeviceContext(id, ditem.device.get());

    params.track_allocations = false;
    stats = nullptr;
    if (m_state->stats_collector_ && !tagged_node.is_dead) {
        // track allocations if and only if we are collecting statistics
        params.track_allocations = true;
        stats = new tf::NodeExecStats;
        stats->set_node_name(node->name());
        nodestats::SetScheduled(stats, scheduled_usec);
        nodestats::SetAllStart(stats);
    }

    DEBUG("Process node: {}, {}", SummarizeNodeDef(node->def()), rctx);

    auto input_tensors = m_state->GetInputTensors(input_frame, input_iter);
    auto first_input = input_tensors + item.input_start;

    ExecutorState::EntryVector outputs; // for use of sync compute
    tf::TensorReferenceVector accessed_tensors;
    tf::DeviceContext *device_context = nullptr;

    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    bool launched_asynchronously = false;
    if (tagged_node.is_dead && !IsTransferNode(node)) {
        outputs.resize(item.num_outputs);
    } else {
        // Prepares inputs.
        bool is_input_dead = false;
        s = m_state->PrepareInputs(item, op_kernel, ditem.device, params.op_device_context,
                        first_input, &inputs, &input_device_contexts, &input_alloc_attrs,
                        &is_input_dead);
        if (!s.ok()) {
            // Clear inputs.
            m_state->ClearInputs(first_input, item.num_inputs);
            m_state->MaybeMarkCompleted(input_frame, input_iter, id);
            afterRun(s, cbs);
            return;
        }

        // Set up compute params.
        params.op_kernel = op_kernel;
        params.frame_iter = tf::FrameAndIter(input_frame->frame_id, input_iter);
        params.is_input_dead = is_input_dead;
        params.output_attr_array = item.output_attrs();

        if (kernel_is_async) {
            // Asynchronous computes.
            TRACE("Launch Async kernel");
            auto async = op_kernel->AsAsync();
            DCHECK(async != nullptr);
            launched_asynchronously = true;

            auto pstate = new ExecutorState::AsyncState(params, tagged_node, &item, first_input, stats);

            // `done` should be called last as `this` would be deleted in it.
            auto asyncDone = [this, pstate, cbs, ditem = ditem, execState = m_state]() {
                auto state = std::unique_ptr<ExecutorState::AsyncState>(pstate);
                auto &device = ditem.device;
                auto stats = state->stats;
                auto first_input = state->first_input;

                // Inspect return state for retrying on memory failure
                if (maybeMemoryFailure(state->ctx.status(), cbs.memFailure)) {
                    return;
                }

                TRACE(" Async kernel done: {}", SummarizeNodeDef(state->item->node->def()));
                if (stats)
                    nodestats::SetOpEnd(stats);
                ExecutorState::EntryVector outputs;
                auto s = execState->ProcessOutputs(*state->item, &state->ctx, *rctx, device, &outputs, stats);
                if (stats)
                    nodestats::SetMemory(stats, &state->ctx);
                // Clears inputs.
                execState->ClearInputs(first_input, state->item->num_inputs);
                // mark completed
                auto input_frame = state->tagged_node.input_frame;
                const int64_t input_iter = state->tagged_node.input_iter;
                const int id = state->tagged_node.node->id();
                execState->MaybeMarkCompleted(input_frame, input_iter, id);
                // propagate outputs
                ExecutorState::TaggedNodeSeq ready;
                if (s.ok()) {
                    execState->PropagateOutputs(state->tagged_node, state->item, &outputs, &ready);
                }
                outputs.clear();
                // record tensor access
                if (s.ok() && ditem.device_record_tensor_access) {
                    // Get the list of all tensors accessed during the execution
                    tf::TensorReferenceVector accessed;
                    state->ctx.retrieve_accessed_tensors(&accessed);
                    if (stats)
                        nodestats::SetReferencedTensors(stats, accessed);
                    // callee takes ownership of the vector
                    device->ConsumeListOfAccessedTensors(state->ctx.op_device_context(), accessed);
                }

                auto completed = execState->NodeDone(s, state->item->node, device.get(), state->params.rendezvous,
                                                     ready, stats, nullptr);

                num_finished_ops.notify();
                if (completed) {
                    execState->Finish();
                }
                cbs.done();
            };
            if (stats)
                nodestats::SetOpStart(stats);
            ditem.device->ComputeAsync(async, &pstate->ctx, std::move(asyncDone));
        } else {
            // Synchronous computes.
            TRACE("Launch sync kernel");
            tf::OpKernelContext ctx(&params, item.num_outputs);
            if (stats)
                nodestats::SetOpStart(stats);
            ditem.device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
            if (stats)
                nodestats::SetOpEnd(stats);

            // Inspect return state for retrying on memory failure
            if (maybeMemoryFailure(ctx.status(), cbs.memFailure)) {
                return;
            }

            TRACE("Sync ProcessOutputs");
            s = m_state->ProcessOutputs(item, &ctx, *rctx, ditem.device, &outputs, stats);
            if (s.ok() && ditem.device_record_tensor_access) {
                // Get the list of all tensors accessed during the execution
                ctx.retrieve_accessed_tensors(&accessed_tensors);
                device_context = ctx.op_device_context();
            }
            if (stats)
                nodestats::SetMemory(stats, &ctx);
        }
    }

    if (!launched_asynchronously) {
        // Clears inputs.
        m_state->ClearInputs(first_input, item.num_inputs);
        m_state->MaybeMarkCompleted(input_frame, input_iter, id);
        // Propagates outputs.
        if (s.ok()) {
            TRACE("Propagates outputs");
            m_state->PropagateOutputs(tagged_node, &item, &outputs, &ready);
        }
        outputs.clear();
        if (!accessed_tensors.empty()) {
            if (stats)
                nodestats::SetReferencedTensors(stats, accessed_tensors);
            // device_context is set above in synchronous computes
            ditem.device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
        }
        if (stats) {
            scheduled_usec = nodestats::NowInUsec();
        }
        // Postprocess.
        afterRun(s, cbs);
        TRACE("Postprocess completed: {}", completed);
    } else {
        cbs.launched();
    }
}

void ExecTask::afterRun(const tf::Status &s, const Callbacks &cbs)
{
    completed = m_state->NodeDone(s, tagged_node.node, ditem.device.get(), params.rendezvous,
                                  ready, stats, &inline_ready);
    num_finished_ops.notify();
    cbs.launched();
    cbs.done();
}

bool ExecTask::maybeMemoryFailure(const tf::Status &s, DoneCallback memFailure)
{
    if (s.code() == tf::error::RESOURCE_EXHAUSTED) {
        // we didn't implement rollback. So this can only happen for non ref input ops
        assert(!has_ref_input);

        ++failureTimes;
        if (memFailure) {
            memFailure();
        }

        return true;
    }
    return false;
}

ExecTask::~ExecTask()
{
    // At this time m_state may already be deleted.
    if (op_kernel) {
        deleteKernel(op_kernel, ditem.function_library.get());
    }
}
