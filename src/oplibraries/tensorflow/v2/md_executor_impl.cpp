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

#include "oplibraries/tensorflow/v2/peropallocdevice.h"

#include "oplibraries/tensorflow/tensorflow_headers.h"

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

namespace {
tf::PartialTensorShape fromShapeHandle(tf::shape_inference::InferenceContext *ctx,
                                       tf::shape_inference::ShapeHandle sph)
{
    if (!ctx->RankKnown(sph)) {
        return {};
    }
    auto rank = ctx->Rank(sph);
    std::vector<tf::int64> vec(rank);
    for (int i = 0; i != rank; ++i) {
        vec[i] = ctx->Value(ctx->Dim(sph, i));
    }
    return tf::PartialTensorShape(vec);
}

/**
 * Only accepts _Send or _Recv nodes
 */
std::string rendezKey(const tf::Node *n, uint64_t frame_id, int64_t iter)
{
    std::string send_device, recv_device, tensor_name;
    tf::int64 send_device_incarnation;
    auto ok = tf::GetNodeAttr(n->def(), "send_device", &send_device);
    if (!ok.ok()) {
        ERR("Node {} doesn't have required attribute: send_device", n->name());
    }
    ok = tf::GetNodeAttr(n->def(), "recv_device", &recv_device);
    if (!ok.ok()) {
        ERR("Node {} doesn't have required attribute: recv_device", n->name());
    }
    ok = tf::GetNodeAttr(n->def(), "send_device_incarnation", &send_device_incarnation);
    if (!ok.ok()) {
        ERR("Node {} doesn't have required attribute: send_device_incarnation", n->name());
    }
    ok = tf::GetNodeAttr(n->def(), "tensor_name", &tensor_name);
    if (!ok.ok()) {
        ERR("Node {} doesn't have required attribute: tensor_name", n->name());
    }

    return tf::strings::StrCat(send_device, ";",
                           tf::strings::FpToString(send_device_incarnation), ";",
                           recv_device, ";",
                           tensor_name, ";",
                           frame_id, ":", iter);
}

} // namespace

void ExecutorState::fetchRecvShape(const tf::Node *n)
{
    if (!IsRecv(n)) {
        return;
    }

    auto zr = static_cast<tf::ZrpcRemoteRendezvous*>(rendezvous_);
    assert(zr);

    auto key = rendezKey(n, 0, 0);

    tf::Tensor t;
    if (zr->FindTensor(key, t)) {
        tf::mutex_lock l(refinerMu_);
        sendShapes_[key] = tf::PartialTensorShape(t.shape().dim_sizes());
    } else {
        WARN("Client terminated recv key not found: {}", key);
    }
}

void ExecutorState::addNodeToRefiner(const TaggedNode &tn)
{
    tf::mutex_lock l(refinerMu_);
    auto node = tn.node;

    auto ok = refiner_.AddNode(node);
    if (!ok.ok()) {
        ERR("Error when adding node {} to shape refiner: {}", node->name(), ok);
    }

    // Special handling for some nodes
    if (node->type_string() == "_Send" || node->type_string() == "_HostSend") {
        // There is only one input
        auto e = *node->in_edges().begin();
        auto ctx = refiner_.GetContext(e->src());
        if (!ctx) {
            ERR("Input '{}' for '{}' was not previously added to ShapeRefiner.",
                e->src()->name(), node->name());
            return;
        }
        auto key = rendezKey(tn.node, tn.input_frame->frame_id, tn.input_iter);
        sendShapes_[key] = fromShapeHandle(ctx, ctx->output(e->src_output()));
    } else if (node->type_string() == "_Recv" || node->type_string() == "_HostRecv") {
        auto key = rendezKey(tn.node, tn.input_frame->frame_id, tn.input_iter);
        auto it = sendShapes_.find(key);
        if (it == sendShapes_.end()) {
            ERR("Send op with key '{}' for '{}' was not previously added to ShapeRefiner.",
                key, node->name());
            return;
        }
        auto &shape = it->second;
        auto ctx = refiner_.GetContext(node);
        // ctx cannot be nullptr because ok.ok()
        const int num_dims = shape.dims();
        if (num_dims < 0) {
            ctx->set_output(0, ctx->UnknownShape());
        } else {
            std::vector<tf::shape_inference::DimensionHandle> dims(num_dims);
            for (int i = 0; i < num_dims; ++i) {
                // -1 is unknown in PartialTensorShape and in InferenceContext, so this size
                // can be passed directly to MakeDim.
                dims[i] = ctx->MakeDim(shape.dim_size(i));
            }
            ctx->set_output(0, ctx->MakeShape(dims));
        }
    }
}

bool ExecutorImpl::handlePagingRequest(uint64_t oldTicket, std::shared_ptr<ResourceContext> &&rctx)
{
    utils::Guard g(entry_mu_);
    auto it = active_entries_.find(oldTicket);
    if (it == active_entries_.end()) {
        ERR("Requested ticket for paging not found: {}", oldTicket);
        return false;
    }

    auto &entry = *it->second;
    if (entry.expect_ref) {
        ERR("Can't move an reference tensor");
        return false;
    }

    // Create target device
    DeviceItem item;
    auto ok = LookupDevice(rctx->spec, &item);
    if (!ok.ok()) {
        ERR("Error when looking up device for paging: {}", ok);
        return false;
    }
    item.device->setResourceContext(std::move(rctx));

    // Move!
    ok = derefMoveTensor(entry, item.device, nullptr, {},
                         tf::strings::StrCat("Paging tensor of ticket ", oldTicket));

    if (!ok.ok()) {
        ERR("Error when paging out tensor: {}", ok);
        return false;
    }
    return true;
}

void ExecutorImpl::forceEvicted(uint64_t ticket, void *addr)
{
    UNUSED(ticket);
    UNUSED(addr);
    // TODO: handle when addr was force evicted.
}

std::unique_ptr<PerOpAllocDevice> ExecutorImpl::CreatePerOpAllocDevice(tf::Device *dev)
{
    // TODO: impliment a free list
    return std::make_unique<PerOpAllocDevice>(dev);
}

tf::Status ExecutorImpl::LookupDevice(const DeviceSpec &spec, DeviceItem *item)
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

    tf::Device *tfdev;
    auto ok = params_.deviceMgr->LookupDevice(name, &tfdev);
    if (!ok.ok()) {
        ERR("Cannot find device for {}: {}", spec, ok);
        return ok;
    }
    item->device = CreatePerOpAllocDevice(tfdev);

    auto fruntime = params_.create_fruntime(item->device.get());
    item->function_library.reset(fruntime, params_.delete_fruntime);

    item->device_record_tensor_access = item->device->RequiresRecordingAccessedTensors();
    return tf::Status::OK();
}
