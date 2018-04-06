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

/*
 * Make sure tensorflow_headers is included first before
 * any other headers, so we can correctly override TF logging
 * with ours.
 */
#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "md_executor_impl.h"

#include "oplibraries/tensorflow/device/salusdevices.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/v2/tfallocator.h"
#include "oplibraries/tensorflow/worker/rendezvousmgr.h"
#include "utils/containerutils.h"
#include "utils/cpp17.h"
#include "utils/stringutils.h"
#include "utils/debugging.h"
#include <boost/iterator/indirect_iterator.hpp>
#include <boost/thread/lock_algorithms.hpp>
#include <unordered_set>
#include <vector>
#include <oplibraries/tensorflow/tfinstance.h>

using std::chrono::duration_cast;
using FpMS = std::chrono::duration<double, std::chrono::milliseconds::period>;
using namespace std::chrono_literals;
using namespace date;

namespace salus::oplib::tensorflow {

/*
namespace nodestats {
void SetScheduled(tf::NodeExecStats *nt, int64_t t)
{
    if (!nt) return;
    nt->set_scheduled_micros(t);
}

void SetAllStart(tf::NodeExecStats *nt)
{
    if (!nt) return;
    nt->set_all_start_micros(NowInUsec());
}

void SetOpStart(tf::NodeExecStats *nt)
{
    if (!nt) return;
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
    for (const auto &tensor : tensors) {
        auto description = nt->add_referenced_tensor();
        tensor.FillDescription(description);
    }
}

// Sets the timeline_label field of *node_stats, using data from *node.
// Returns true iff the node is a transfer node.
// TODO(tucker): merge with the DetailText function in session.cc
// in a common location.
bool SetTimelineLabel(tf::NodeExecStats *node_stats, const tf::Node *node)
{
    bool is_transfer_node = false;
    std::string memory;
    for (auto &all : node_stats->memory()) {
        auto tot = all.total_bytes();
        if (tot >= 0.1 * 1048576.0) {
            auto peak = all.peak_bytes();
            if (peak > 0) {
                memory = tf::strings::StrCat(memory, "[", all.allocator_name(),
                                             tf::strings::Printf(" %.1fMB %.1fMB] ", tot / 1048576.0,
                                                                 peak / 1048576.0));
            } else {
                memory = tf::strings::StrCat(memory, "[", all.allocator_name(),
                                             tf::strings::Printf(" %.1fMB] ", tot / 1048576.0));
            }
        }
    }
    auto def = node->def();
    std::string text;
    if (IsSend(node)) {
        std::string tensor_name;
        TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
        std::string recv_device;
        TF_CHECK_OK(GetNodeAttr(def, "recv_device", &recv_device));
        text = tf::strings::StrCat(memory, def.name(), " = ", def.op(), "(", tensor_name, " @", recv_device);
        is_transfer_node = true;
    } else if (IsRecv(node)) {
        std::string tensor_name;
        TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
        std::string send_device;
        TF_CHECK_OK(GetNodeAttr(def, "send_device", &send_device));
        text = tf::strings::StrCat(memory, def.name(), " = ", def.op(), "(", tensor_name, " @", send_device);
        is_transfer_node = true;
    } else {
        text = tf::strings::StrCat(
            memory, def.name(), " = ", def.op(), "(",
            tf::str_util::Join(std::vector<tf::StringPiece>(def.input().begin(), def.input().end()), ", "),
            ")");
    }
    node_stats->set_timeline_label(text);
    return is_transfer_node;
}
} // namespace nodestats
*/

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
        LOG(ERROR) << "Node " << n->name() << " doesn't have required attribute: send_device";
    }
    ok = tf::GetNodeAttr(n->def(), "recv_device", &recv_device);
    if (!ok.ok()) {
        LOG(ERROR) << "Node " << n->name() << " doesn't have required attribute: recv_device";
    }
    ok = tf::GetNodeAttr(n->def(), "send_device_incarnation", &send_device_incarnation);
    if (!ok.ok()) {
        LOG(ERROR) << "Node " << n->name() << " doesn't have required attribute: send_device_incarnation";
    }
    ok = tf::GetNodeAttr(n->def(), "tensor_name", &tensor_name);
    if (!ok.ok()) {
        LOG(ERROR) << "Node " << n->name() << " doesn't have required attribute: tensor_name";
    }

    return tf::strings::StrCat(send_device, ";", tf::strings::FpToString(send_device_incarnation), ";",
                               recv_device, ";", tensor_name, ";", frame_id, ":", iter);
}

} // namespace

void ExecutorState::fetchRecvShape(const tf::Node *n)
{
    if (!IsRecv(n)) {
        return;
    }

    auto zr = static_cast<WorkerRendezvous *>(rendezvous_); // NOLINT
    DCHECK_NOTNULL(zr);

    auto key = rendezKey(n, 0, 0);

    tf::Tensor t;
    if (zr->FindTensor(key, t)) {
        sstl::Guard l(refinerMu_);
        sendShapes_[key] = tf::PartialTensorShape(t.shape().dim_sizes());
    } else {
        VLOG(2) << "Recv key not found for a client terminated recv op : " << key;
    }
}

void ExecutorState::addNodeToRefiner(const TaggedNode &tn)
{
    sstl::Guard l(refinerMu_);
    auto node = tn.node;

    if (node->type_string() == "Slice") {
        VLOG(2) << "Skipping node " << node->name() << " with bugous shape_fn";
        return;
    }

    auto ok = refiner_.AddNode(node);
    if (!ok.ok()) {
        VLOG(3) << "Error when adding node " << node->name() << " to shape refiner: " << ok;
    }

    // Special handling for some nodes
    if (node->type_string() == "_Send" || node->type_string() == "_HostSend") {
        // There is only one input
        auto e = *node->in_edges().begin();
        auto ctx = refiner_.GetContext(e->src());
        if (!ctx) {
            VLOG(3) << "Input '" << e->src()->name() << "' for '" << node->name()
                    << "' was not previously added to ShapeRefiner.";
            return;
        }
        auto key = rendezKey(tn.node, tn.input_frame->frame_id, tn.input_iter);
        sendShapes_[key] = fromShapeHandle(ctx, ctx->output(e->src_output()));
    } else if (node->type_string() == "_Recv" || node->type_string() == "_HostRecv") {
        auto key = rendezKey(tn.node, tn.input_frame->frame_id, tn.input_iter);
        auto it = sendShapes_.find(key);
        if (it == sendShapes_.end()) {
            VLOG(3) << "Send op with key '" << key << "' for '" << node->name()
                    << "' was not previously added to ShapeRefiner.";
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

size_t ExecutorImpl::handlePagingRequest(uint64_t oldTicket, std::unique_ptr<ResourceContext> &&rctx)
{
    // There may be multiple tensor entries that uses this ticket,
    // and potentially share the storage.
    // We want to move one complete set of tensors that are sharing buffer.
    size_t totalReleased = 0;
    std::vector<TensorBufferTree *> parts;
    parts.reserve(4);
    BufferMutexSet reflocks;
    reflocks.reserve(8);

    // guard after decl of parts, because we need to use it.
    sstl::ScopeGuards sg;
    sg += [&totalReleased]() { VLOG(2) << "Paging released " << totalReleased << " bytes of memory"; };

    {
        sstl::Guard g(entry_mu_);
        auto range = active_buffers_.equal_range(oldTicket);
        if (range.first == range.second) {
            LOG(ERROR) << "Requested ticket for paging not found: " << oldTicket;
            return 0;
        }
        // Take candidate parts out of active_buffers_
        auto it = range.first;
        while (it != range.second) {
            auto &tree = it->second;
            DCHECK(tree);
            if (!tree->paged_out && !tree->empty() && tree->root_buf) {
                // candidate
                reflocks.insert(&tree->buf_mu);
                parts.push_back(tree);

                VLOG(2) << "Removing tree " << as_hex(tree) << " of ticket " << oldTicket << " due to paging";
                it = active_buffers_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Remember to add them back to active entries with updated value when exit
    sg += [this, &parts, oldTicket]() {
        sstl::Guard g(entry_mu_);
        for (auto part : parts) {
            VLOG(2) << "Adding buffer tree of ticket " << part->ticket << " (was " << oldTicket
                    << ") due to paging";
            active_buffers_.emplace(part->ticket, part);
        }
    };

    if (parts.empty()) {
        LOG(WARNING) << "No tensor available for paging";
        return totalReleased;
    }

    // Create target device
    DeviceItem item;
    auto ok = LookupDevice(rctx->spec(), std::move(rctx), &item);
    if (!ok.ok()) {
        LOG(ERROR) << "Error when looking up device for paging: " << ok;
        return totalReleased;
    }

    // Lock all buffer, and all read/write should happen after this
    sstl::lock(reflocks.begin(), reflocks.end());
    std::vector<std::unique_lock<boost::upgrade_mutex>> guards;
    guards.reserve(reflocks.size());
    for (auto l : reflocks) {
        guards.emplace_back(*l, std::adopt_lock);
    }

    for (auto &part : parts) {
        DCHECK(!part->paged_out);
        if (!part->root_buf) {
            // We use empty root_buf as a dumy tree for
            // uninitialized tensors, although it's unlikely
            // this particular tree get paged out, we have to
            // handle this case
            continue;
        }

        auto oldRoot = part->root_buf;
        auto su = sstl::add_ref(oldRoot);

        auto size = oldRoot->size();
        auto ok = moveTensorTree(*part, item.device);
        if (ok.ok()) {
            DCHECK(oldRoot->RefCountIsOne());
            VLOG(2) << "Releasing old root buffer " << as_hex(oldRoot) << " with data block at "
                    << as_hex(oldRoot->data()) << " of size " << size;

            totalReleased += size;
            part->paged_out = true;
        } else {
            LOG(ERROR) << "Failed to moveTensorTree when paging ticket " << oldTicket << ": " << ok;
        }
    }

    return totalReleased;
}

void ExecutorImpl::forceEvicted()
{
    sstl::Guard g(entry_mu_);
    for (auto state : active_states_) {
        state->ForceInterrupt(tf::errors::ResourceExhausted("Forcely killed due to paging"));
    }
    active_states_.clear();
}

tf::Status ExecutorImpl::LookupTFDevice(const DeviceSpec &spec, tf::Device **tfdev)
{
    *tfdev = TFInstance::instance().tfdevice(spec);
    if (*tfdev) {
        return tf::Status::OK();
    }
    return tf::errors::InvalidArgument("Cannot find device for ", spec.debugString());
}

tf::Status ExecutorImpl::LookupDevice(const DeviceSpec &spec, std::unique_ptr<ResourceContext> &&rctx,
                                      DeviceItem *item)
{
    sstl::TimeoutWarning tw(2ms, [&](auto limit, auto dur){
        LOG(WARNING) << "LookupDevice took more than " << duration_cast<FpMS>(limit)
                     << " to finish: " << duration_cast<FpMS>(dur)
                     << " for " << spec;
    });

    tf::Device *tfdev = nullptr;
    auto ok = LookupTFDevice(spec, &tfdev);
    if (!ok.ok()) {
        return ok;
    }

    auto sdev = ISalusDevice::safe_cast(tfdev);
    if (!sdev) {
        ok = tf::errors::Internal(
                   tf::strings::StrCat("Device is not an ISalusDevice: ", spec.debugString()));
        return ok;
    }

    item->device = sdev->createPerTaskDevice(graph_.get(), std::move(rctx));

    item->function_library = params_.create_fruntime(item->device.get());

    item->device_record_tensor_access = item->device->RequiresRecordingAccessedTensors();
    return tf::Status::OK();
}

POpKernel ExecutorImpl::SetupKernel(sstl::not_null<const tf::Node *> node, const DeviceItem &ditem)
{
    sstl::TimeoutWarning tw(2ms, [&](auto limit, auto dur){
        LOG(WARNING) << "SetupKernel took more than " << duration_cast<FpMS>(limit)
                     << " to finish: " << duration_cast<FpMS>(dur)
                     << " for " << node->name() << " on " << ditem.device->name();
    });
    // first check if we have a cache for this kernel and if so, if the kernel is on the same device
    {
        sstl::Guard g(kernel_dev_mu_);
        auto it = kernel_dev_.find(node->name());
        if (it != kernel_dev_.end())
            if (it->second != &ditem.device->underlayingDevice()) {
                throw TFException(
                    tf::errors::AlreadyExists("Kernel previously created on another device:", node->name(),
                                              " previous device: ", it->second->name(), " requested device: ",
                                              ditem.device->underlayingDevice().name()));
            }
    }

    VLOG(2) << "Creating a kernel for device: " << ditem.device->name();
    auto popkernel = params_.get_kernel(node->def(), ditem.function_library.get());

    // only record device placement after create kernel, because get_kernel may throw
    {
        sstl::Guard g(kernel_dev_mu_);
        kernel_dev_.emplace(node->name(), &ditem.device->underlayingDevice());
    }
    return popkernel;
}

/**
 * If entry->alloc_tree is not nullptr, add entry to entry->alloc_tree
 * Else, find/create tree based on root_buf, and add entry to the tree
 * When the entry contains an uninitialized tensor, an special tree is
 * assigned.
 * TODO: currently the special tree for uninitialized tensor is create
 * per ticket. This can actually be a global static
 */
void ExecutorImpl::updateBufferTree(Entry *entry, uint64_t ticket)
{
    DCHECK(entry);
    DCHECK(entry->has_value);

    const auto buf = tf::remote::PagingHelper::bufferOf(*entry->RefOrVal());
    const auto root_buf = buf ? buf->root_buffer() : nullptr;

    sstl::Guard g(entry_mu_);
    auto &tree = entry->alloc_tree;
    if (!tree) {
        auto range = active_buffers_.equal_range(ticket);
        for (auto it = range.first; it != range.second; ++it) {
            DCHECK(it->second);
            if (it->second->root_buf == root_buf) {
                tree = it->second;
                break;
            }
        }
        if (!tree) {
            // construct new tree
            tree = new TensorBufferTree;
            buffer_trees_.push_back(*tree);
            active_buffers_.emplace(ticket, tree);

            tree->ticket = ticket;
            tree->root_buf = root_buf;
        }
    }
    DCHECK(tree);
    DCHECK_EQ(tree->ticket, ticket);
    DCHECK_EQ(tree->root_buf, root_buf);

    VLOG(2) << "Adding entry " << as_hex(entry) << " to tree " << tree << " of buffer " << tree->root_buf
            << " with ticket " << ticket;

    bool added = false;
    if (root_buf == buf) {
        auto it = std::find(tree->roots.begin(), tree->roots.end(), entry);
        if (it == tree->roots.end()) {
            added = true;
            tree->roots.emplace_back(entry);
        }
    } else {
        auto &sub = tree->subs[buf];
        auto it = std::find(sub.begin(), sub.end(), entry);
        if (it == sub.end()) {
            added = true;
            sub.emplace_back(entry);
        }
    }

    if (added && root_buf) {
        root_buf->Ref();
    }
}

void ExecutorImpl::removeFromBufferTree(const Entry *entry, EntryVec *needUpdate)
{
    DCHECK(entry);

    auto tree = entry->alloc_tree;
    if (!tree) {
        return;
    }

    auto matchRefs = [needUpdate, entry](auto e) {
        if (e == entry || (needUpdate && entry->ref && e->ref == entry->ref)) {
            VLOG(2) << "Removing entry " << as_hex(e) << " from tree " << entry->alloc_tree << " of buffer "
                    << entry->alloc_tree->root_buf << " with ticket " << entry->alloc_tree->ticket;
            if (entry->alloc_tree->root_buf) {
                entry->alloc_tree->root_buf->Unref();
            }
            e->alloc_tree = nullptr;
            if (needUpdate) {
                needUpdate->push_back(e);
            }
            return true;
        }
        return false;
    };

    sstl::Guard g(entry_mu_);

    if (sstl::erase_if(tree->roots, matchRefs)) {
        return;
    }
    // the entry was not found in roots, so it must be in one of the subs
    for (auto &p : tree->subs) {
        if (sstl::erase_if(p.second, matchRefs)) {
            break;
        }
    }
}

} // namespace salus::oplib::tensorflow
