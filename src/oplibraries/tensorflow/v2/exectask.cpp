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
#include "oplibraries/tensorflow/v2/md_rendezvous.h"

#include "oplibraries/tensorflow/tensorflow_headers.h"

namespace tf = tensorflow;

ExecTask::ExecTask(ExecutorState *state, utils::semaphore *se,
                   ExecutorState::TaggedNode &node, ExecutorState::TaggedNodeSeq &ready,
                   ExecutorState::TaggedNodeReadyQueue &inline_ready,
                   tf::NodeExecStats *stats, tf::OpKernelContext::Params &params,
                   int64_t &scheduled_usec, ExecutorState::EntryVector &outputs,
                   TensorValueVec &inputs, DeviceContextVec &input_device_contexts,
                   AllocatorAttributeVec &input_alloc_attrs, bool &completed, tf::Rendezvous *rendez)
    : tagged_node(node)
    , ready(ready)
    , inline_ready(inline_ready)
    , stats(stats)
    , params(params)
    , scheduled_usec(scheduled_usec)
    , outputs(outputs)
    , inputs(inputs)
    , input_device_contexts(input_device_contexts)
    , input_alloc_attrs(input_alloc_attrs)
    , completed(completed)
    , rendez(rendez)
    , m_se(se)
    , m_state(state)
{
}

bool ExecTask::prepare(DeviceSpec &dev)
{
    auto s = LookupDevice(dev, ditem);
    if (!s.ok()) {
        return false;
    }

    // Instantiate kernel
    op_kernel = nullptr;
    s = m_state->SetupKernel(tagged_node, ditem, &op_kernel);
    if (!s.ok()) {
        return false;
    }

    CHECK(op_kernel);
    kernel_is_async = (op_kernel->AsAsync() != nullptr);

    return true;
}

tf::Status ExecTask::LookupDevice(const DeviceSpec &spec, DeviceItem &item)
{
    std::string name;
    switch (spec.type) {
    case DeviceType::CPU:
        name = "CPU:";
        break;
    case DeviceType::GPU:
        name = "GPU:";
        break;
    default:
        name = "CPU:";
        break;
    }
    name += std::to_string(spec.id);

    auto ok = m_state->impl_->params_.deviceMgr->LookupDevice(name, &item.device);
    if (!ok.ok()) {
        ERR("Cannot find device for {}: {}", spec, ok);
        return ok;
    }
    item.function_library = m_state->FindFunctionLibrary(item.device);
    item.device_record_tensor_access = item.device->RequiresRecordingAccessedTensors();
    return tf::Status::OK();
}

ProtoPtr ExecTask::run()
{
    const auto &gview = m_state->impl_->gview_;
    auto node = tagged_node.node;
    auto input_frame = tagged_node.input_frame;
    int64_t input_iter = tagged_node.input_iter;
    const size_t id = node->id();
    const auto &item = *gview.node(id);

    auto s = gview.SetAllocAttrForNode(node, ditem.device, op_kernel);
    if (!s.ok()) {
        m_state->MaybeMarkCompleted(input_frame, input_iter, id);
        // Continue to process the nodes in 'inline_ready'.
        completed = m_state->NodeDone(s, item.node, ditem.device, nullptr, ready, stats, &inline_ready);
        m_se->notify();
        return {};
    }

    params.device = ditem.device;
    auto localRendez = new MultiDeviceRendezvous(ditem.device, rendez);
    params.rendezvous = localRendez;
    params.record_tensor_accesses = ditem.device_record_tensor_access;
    params.function_library = ditem.function_library;
    // Set the device_context for this node id, if it exists.
    params.op_device_context = m_state->FindDeviceContext(id, ditem.device);

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

    DEBUG("Process node: {} step {} {} is dead {}: on device {}",
          id, params.step_id, SummarizeNodeDef(node->def()), tagged_node.is_dead, ditem.device->name());

    auto input_tensors = m_state->GetInputTensors(input_frame, input_iter);
    auto first_input = input_tensors + item.input_start;
    outputs.clear();

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
            int num_inputs = item.num_inputs;
            for (int i = 0; i < num_inputs; ++i) {
                (first_input + i)->ClearVal();
            }
            m_state->MaybeMarkCompleted(input_frame, input_iter, id);
            // Continue to process the nodes in 'inline_ready'.
            completed = m_state->NodeDone(s, item.node, ditem.device, localRendez, ready, stats, &inline_ready);
            m_se->notify();
            return {};
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
            auto state = new ExecutorState::AsyncState(params, tagged_node, &item, first_input, stats);

            // Don't capture `this`, as when the cb get called, the task may already be deleted
            auto done = [state, localRendez, ditem = ditem, execState = m_state]() {
                auto device = ditem.device;
                auto stats = state->stats;     // Shorthand
                auto first_input = state->first_input; // Shorthand

                TRACE(" Async kernel done: {}", SummarizeNodeDef(state->item->node->def()));
                if (stats)
                    nodestats::SetOpEnd(stats);
                ExecutorState::EntryVector outputs;
                tf::Status s = execState->ProcessOutputs(*state->item, &state->ctx, device,
                                                         &outputs, stats);
                if (stats)
                    nodestats::SetMemory(stats, &state->ctx);
                // Clears inputs.
                const int num_inputs = state->item->num_inputs;
                for (int i = 0; i < num_inputs; ++i) {
                    (first_input + i)->ClearVal();
                }
                auto input_frame = state->tagged_node.input_frame;
                const int64_t input_iter = state->tagged_node.input_iter;
                const int id = state->tagged_node.node->id();
                execState->MaybeMarkCompleted(input_frame, input_iter, id);
                ExecutorState::TaggedNodeSeq ready;
                if (s.ok()) {
                    execState->PropagateOutputs(state->tagged_node, state->item, &outputs, &ready);
                }
                outputs.clear();
                if (s.ok() && ditem.device_record_tensor_access) {
                    // Get the list of all tensors accessed during the execution
                    tf::TensorReferenceVector accessed;
                    state->ctx.retrieve_accessed_tensors(&accessed);
                    if (stats)
                        nodestats::SetReferencedTensors(stats, accessed);
                    // callee takes ownership of the vector
                    device->ConsumeListOfAccessedTensors(state->ctx.op_device_context(), accessed);
                }
                bool completed = execState->NodeDone(s, state->item->node, device, localRendez, ready, stats, nullptr);
                delete state;
                if (completed)
                    execState->Finish();
            };
            if (stats)
                nodestats::SetOpStart(stats);
            ditem.device->ComputeAsync(async, &state->ctx, done);
        } else {
            // Synchronous computes.
            TRACE("Launch sync kernel");
            tf::OpKernelContext ctx(&params, item.num_outputs);
            if (stats)
                nodestats::SetOpStart(stats);
            ditem.device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
            if (stats)
                nodestats::SetOpEnd(stats);

            TRACE("Sync ProcessOutputs");
            s = m_state->ProcessOutputs(item, &ctx, ditem.device, &outputs, stats);
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
        const int num_inputs = item.num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->ClearVal();
        }
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
        completed = m_state->NodeDone(s, item.node, ditem.device, localRendez, ready, stats, &inline_ready);
        TRACE("Postprocess completed: {}", completed);
    }
    m_se->notify();
    return {};
}

ExecTask::~ExecTask() = default;