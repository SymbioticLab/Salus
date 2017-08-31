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

#include "md_executor_impl.h"

#include <tensorflow/core/lib/strings/stringprintf.h>

namespace nodestats {
void SetScheduled(tf::NodeExecStats *nt, int64_t t)
{
    nt->set_scheduled_micros(t);
}

void SetAllStart(tf::NodeExecStats *nt)
{
    nt->set_all_start_micros(NowInUsec());
}

void SetOpStart(tf::NodeExecStats *nt)
{
    DCHECK_NE(nt->all_start_micros(), 0);
    nt->set_op_start_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOpEnd(tf::NodeExecStats *nt)
{
    DCHECK_NE(nt->all_start_micros(), 0);
    nt->set_op_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetAllEnd(tf::NodeExecStats *nt)
{
    DCHECK_NE(nt->all_start_micros(), 0);
    nt->set_all_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOutput(tf::NodeExecStats *nt, int slot, const tf::Tensor *v)
{
    DCHECK(v);
    auto no = nt->add_output();
    no->set_slot(slot);
    v->FillDescription(no->mutable_tensor_description());
}

void SetMemory(tf::NodeExecStats *nt, tf::OpKernelContext *ctx)
{
    for (const auto &allocator_pair : ctx->wrapped_allocators()) {
        auto memory = nt->add_memory();
        // retrieving the sizes from the wrapped allocator removes the
        // executor's reference to it, so allocator_pair.second must not
        // be dereferenced again after this statement
        auto sizes = allocator_pair.second->GetSizesAndUnRef();
        memory->set_allocator_name(allocator_pair.first->Name());
        memory->set_total_bytes(std::get<0>(sizes));
        if (allocator_pair.first->TracksAllocationSizes()) {
            memory->set_peak_bytes(std::get<1>(sizes));
            memory->set_live_bytes(std::get<2>(sizes));
        }
    }
    auto *ms = nt->mutable_memory_stats();
    ms->set_host_temp_memory_size(ctx->host_temp_memory_size());
    ms->set_device_temp_memory_size(ctx->device_temp_memory_size());
    for (const auto &alloc_id : ctx->host_persistent_alloc_ids()) {
        ms->mutable_host_persistent_tensor_alloc_ids()->Add(alloc_id);
    }
    for (const auto &alloc_id : ctx->device_persistent_alloc_ids()) {
        ms->mutable_device_persistent_tensor_alloc_ids()->Add(alloc_id);
    }
    ms->set_host_persistent_memory_size(ctx->host_persistent_memory_allocated());
    ms->set_device_persistent_memory_size(ctx->device_persistent_memory_allocated());
}

void SetReferencedTensors(tf::NodeExecStats *nt, const tf::TensorReferenceVector &tensors)
{
    // be careful not to increment the reference count on any tensor
    // while recording the information
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto description = nt->add_referenced_tensor();
        tensors.at(i).FillDescription(description);
    }
}

// Sets the timeline_label field of *node_stats, using data from *node.
// Returns true iff the node is a transfer node.
// TODO(tucker): merge with the DetailText function in session.cc
// in a common location.
bool SetTimelineLabel(tf::NodeExecStats *node_stats, const tf::Node* node)
{
    bool is_transfer_node = false;
    std::string memory;
    for (auto& all : node_stats->memory()) {
        auto tot = all.total_bytes();
        if (tot >= 0.1 * 1048576.0) {
            auto peak = all.peak_bytes();
            if (peak > 0) {
                memory =
                tf::strings::StrCat(memory, "[", all.allocator_name(),
                                tf::strings::Printf(" %.1fMB %.1fMB] ", tot / 1048576.0,
                                                peak / 1048576.0));
            } else {
                memory = tf::strings::StrCat(memory, "[", all.allocator_name(),
                                         tf::strings::Printf(" %.1fMB] ", tot / 1048576.0));
            }
        }
    }
    auto def = node->def();
    std::string text = "";
    if (IsSend(node)) {
        std::string tensor_name;
        TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
        std::string recv_device;
        TF_CHECK_OK(GetNodeAttr(def, "recv_device", &recv_device));
        text = tf::strings::StrCat(memory, def.name(), " = ", def.op(), "(",
                               tensor_name, " @", recv_device);
        is_transfer_node = true;
    } else if (IsRecv(node)) {
        std::string tensor_name;
        TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
        std::string send_device;
        TF_CHECK_OK(GetNodeAttr(def, "send_device", &send_device));
        text = tf::strings::StrCat(memory, def.name(), " = ", def.op(), "(",
                               tensor_name, " @", send_device);
        is_transfer_node = true;
    } else {
        text = tf::strings::StrCat(
            memory, def.name(), " = ", def.op(), "(",
                               tf::str_util::Join(
                                   std::vector<tf::StringPiece>(def.input().begin(), def.input().end()),
                                              ", "),
                               ")");
    }
    node_stats->set_timeline_label(text);
    return is_transfer_node;
}
} // namespace nodestats
