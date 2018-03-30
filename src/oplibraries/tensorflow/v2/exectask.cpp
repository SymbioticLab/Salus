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
#include "exectask.h"
#include "execution/devices.h"
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "oplibraries/tensorflow/v2/tfallocator.h"
#include "oplibraries/tensorflow/worker/devicecontextwithdevice.h"
#include "oplibraries/tensorflow/worker/rendezvouswithhook.h"
#include "utils/macros.h"
#include "utils/date.h"
#include "utils/threadutils.h"
#include <sstream>

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
using FpSeconds = std::chrono::duration<double, seconds::period>;
using namespace std::chrono_literals;
using namespace date;


namespace salus::oplib::tensorflow {

ExecTask::ExecTask(ExecutorState *state, sstl::semaphore &num_finished_ops,
                   const ExecutorState::TaggedNode &node, const tf::OpKernelContext::Params &initial_params,
                   tf::Rendezvous *rendez, int maxFailures)
    : deleteKernel(state->impl_->params_.delete_kernel)
    , maxFailures(maxFailures)
    , kernel_is_async(false)
    , has_ref_input(false)
    , tagged_node(node)
    , params(initial_params)
    , rendez(rendez)
    , num_finished_ops(num_finished_ops)
    , m_state(state)
{
    params.inputs = &inputs;
    params.input_device_contexts = &input_device_contexts;
    params.input_alloc_attrs = &input_alloc_attrs;

    tf::DeviceTypeVector tftypes;
    auto ok =
        tf::SupportedDeviceTypesForNode({tf::DEVICE_GPU, tf::DEVICE_CPU}, tagged_node.node->def(), &tftypes);
    if (!ok.ok()) {
        LOG(ERROR) << "Error while querying supported device for node " << tagged_node.node->name() << ": "
                   << ok;
    }

    VLOG(2) << "Op " << tagged_node.node->def() << " supports device:";
    supportedTypes.reserve(tftypes.size());
    for (const auto &tft : tftypes) {
        supportedTypes.emplace_back(tfDeviceTypeToType(tft));
        VLOG(2) << "    " << tft.type();
    }

    auto device = tagged_node.node->def().device();
    if (!device.empty()) {
        supportedTypes.clear();
        supportedTypes.push_back(tfDeviceNameToSpec(device).type);
    }
    // pre compute estimated usage
    for (auto t : supportedTypes) {
        inferUsage(t);
    }
}

const std::vector<DeviceType> &ExecTask::supportedDeviceTypes() const
{
    return supportedTypes;
}

bool ExecTask::prepare(std::unique_ptr<ResourceContext> &&rctx)
{
    statusInPrepare = tf::Status::OK();

    auto &spec = rctx->spec();

    auto match = [&spec](auto type) { return type == spec.type; };
    if (std::none_of(supportedTypes.begin(), supportedTypes.end(), match)) {
        return false;
    }

    auto s = m_state->impl_->LookupDevice(spec, std::move(rctx), &ditem);
    if (!s.ok()) {
        statusInPrepare.Update(s);
        return false;
    }

    // First check if we already created the kernel on some device
    op_kernel = nullptr;
    owns_kernel = true;
    std::string devName;
    auto ok = m_state->impl_->params_.find_kernel(tagged_node.node->def(), &devName, &op_kernel);

    bool done = true;
    if (ok.ok() && op_kernel) {
        // we saw this kernel before, check if the device match
        if (devName.empty()) {
            LOG(WARNING) << "We've created the kernel, but don't remember its device: "
                         << tagged_node.node->name();
            auto s = tf::errors::Internal("We've created the kernel, but don't remember its device");
            op_kernel = nullptr;
            done = false;
        } else if (devName != ditem.device->name()) {
            VLOG(3) << "Stateful kernel can not be moved: previously created on " << devName
                    << ", now requested on " << ditem.device->name();
            op_kernel = nullptr;
            done = false;
        }
        // We are on the same device, good.
        // this is a cached kernel.
        owns_kernel = false;
    } else if (!ok.ok()) {
        LOG(ERROR) << "Failed to find kernel with status " << ok << " for Node: " << tagged_node.node->name();
        statusInPrepare.Update(ok);
        // it is okay, just continue to create the kernel
    }

    // Instantiate kernel if not already done
    if (!op_kernel) {
        auto s = m_state->SetupKernel(tagged_node, ditem, &op_kernel);
        if (!s.ok()) {
            LOG(ERROR) << "Error when creating kernel for node " << tagged_node.node->name() << ": " << s;
            statusInPrepare.Update(s);
            done = false;
        }
        // HACK:
        if (op_kernel) {
            owns_kernel = !ditem.function_library->IsStateful(op_kernel->type_string());
        }
    }

    if (!done) {
        // Release the resource context and the device we've given
        ditem.device.reset();
        return done;
    }

    AllocLog(INFO) << "Pre allocated " << ditem.device->resourceContext() << " for " << DebugString();
    kernel_is_async = (op_kernel->AsAsync() != nullptr);

    return true;
}

bool ExecTask::isAsync() const
{
    return kernel_is_async;
}

Resources ExecTask::estimatedUsage(const DeviceSpec &dev)
{
    // First see if we have usage for this node in session
    // but only if we haven't failed before, otherwise,
    // the session cached usage maybe be just a lucky case
    auto usage = m_state->impl_->cachedUsageForNode(tagged_node.node->name());
    if (usage && failureTimes == 0) {
        return *usage;
    }

    // Short-cut if this task has failed before
    if (failureTimes > 0) {
        if (!failedAlloc.empty()) {
            // we don't care if operator[] inserts a new element or not.
            // it it does insert a new element, this means no estimation is available
            // anyway.
            auto usage = cachedUsage[dev];
            resources::merge(usage, failedAlloc);
            return usage;
        }

        const auto &sessHandle = m_state->impl_->params_.session;
        auto rm = m_state->impl_->params_.ins.offeredSessionResource();
        if (rm) {
            auto f = failureTimes;
            if (f > maxFailures) {
                LOG(WARNING) << "Failure time exceeds maximum: " << f << " max " << maxFailures;
                f = maxFailures;
                LOG(WARNING) << "Estimated usage this time: "
                             << resources::DebugString(rm->temporary, "    ");
            }
            uint64_t scale = 1u << (maxFailures - f);
            resources::scale(rm->temporary, 1.0 / scale);

            // Update cache
            cachedUsage[dev] = rm->temporary;
        } else {
            LOG(ERROR) << "No session usage found for exec task: " << tagged_node.node->name()
                       << " under session " << sessHandle;
            // fallback to normal estimation
        }
    }

    // Fast path from cache
    auto it = cachedUsage.find(dev);
    if (it == cachedUsage.end()) {
        inferUsage(dev);
    }
    return cachedUsage[dev];
}

void ExecTask::inferUsage(const DeviceSpec &dev)
{
    // Slow path to calculate the usage
    auto &res = cachedUsage[dev];

    const auto *node = tagged_node.node;
    auto ctx = m_state->shapeForNode(node);
    if (!ctx) {
        LOG(WARNING) << "Shape information not available for node: " << node->name();
        return;
    }

    ExecutorImpl::DeviceItem ditem;
    tf::MemoryTypeVector input_mtypes;
    tf::MemoryTypeVector output_mtypes;
    tf::Device *tfdev;
    auto mtypeStatus = m_state->impl_->LookupTFDevice(dev, &tfdev);
    if (mtypeStatus.ok()) {
        mtypeStatus.Update(tf::remote::MemoryTypesForNode(m_state->impl_->graph_->op_registry(),
                                                          tf::DeviceType(tfdev->device_type()), node->def(),
                                                          &input_mtypes, &output_mtypes));
    }
    if (!mtypeStatus.ok()) {
        LOG(WARNING) << "Kernel not found on device " << dev << ", resource estimation may be inaccurate.";
    }

    ResourceTag devTag{ResourceType::MEMORY, dev};
    ResourceTag cpuTag{ResourceType::MEMORY, dev};

    for (int i = 0; i != ctx->num_outputs(); ++i) {
        auto shp = ctx->output(i);
        if (!ctx->RankKnown(shp)) {
            VLOG(3) << i << "-th output of node " << node->name() << " has unknown rank";
            continue;
        }
        VLOG(3) << "Shape of " << i << "-th output of node " << node->name();
        size_t count = 1;
        for (int j = 0; j != ctx->Rank(shp); ++j) {
            auto dim = ctx->Dim(shp, j);
            if (!ctx->ValueKnown(dim)) {
                VLOG(3) << "    Unknown";
                continue;
            }

            auto val = ctx->Value(dim);
            VLOG(3) << "    " << val;
            count *= val;
        }
        auto dtype = node->output_type(i);
        VLOG(3) << "    dtype " << tf::DataType_Name(dtype) << ", " << tf::DataTypeSize(dtype) << " bytes";
        double subtotal = count * tf::DataTypeSize(dtype);

        if (mtypeStatus.ok() && output_mtypes[i] == tf::HOST_MEMORY) {
            res[cpuTag] += subtotal;
        } else {
            res[devTag] += subtotal;
        }
    }
}

std::string ExecTask::DebugString()
{
    std::ostringstream oss;
    oss << "ExecTask(name=" << tagged_node.node->name() << ", type=" << tagged_node.node->op_def().name()
        << ", session=" << m_state->impl_->params_.session << ", step_id=" << m_state->step_id_
        << ", failures=" << failureTimes << ", inputsize=" << input_size << ")";
    return oss.str();
}

void ExecTask::cancel()
{
    m_state->MaybeMarkCompleted(tagged_node.input_frame, tagged_node.input_iter, tagged_node.node->id());

    auto s = tf::errors::Cancelled("Cancelled");
    // cancel may be called before prepare, so no device
    auto completed = m_state->NodeDone(s, nullptr, nullptr, params.rendezvous, ready);

    num_finished_ops.notify();

    // Do this after cbs.done, because m_state may be accessed in cbs.done
    if (completed) {
        // `m_state` may be deleted in Finish
        m_state->Finish();
    }
}

void ExecTask::run(Callbacks cbs)
{
    const auto &gview = m_state->impl_->gview_;
    auto *node = tagged_node.node;
    const auto input_frame = tagged_node.input_frame;
    const auto input_iter = tagged_node.input_iter;
    const auto id = node->id();
    const auto &item = *gview.node(id);

    // clear early
    params.rendezvous = nullptr;

    if (!statusInPrepare.ok()) {
        m_state->MaybeMarkCompleted(input_frame, input_iter, id);
        afterRun(statusInPrepare, cbs);
        return;
    }

    DCHECK(op_kernel);
    DCHECK(ditem.device);

    // Start run
    auto s = gview.SetAllocAttrForNode(node, ditem.device.get(), op_kernel);
    if (!s.ok()) {
        m_state->MaybeMarkCompleted(input_frame, input_iter, id);
        afterRun(s, cbs);
        return;
    }

    params.device = ditem.device.get();

    auto localRendez = new RendezvousWithHook(ditem.device, sstl::add_ref(rendez));
    params.rendezvous = localRendez;
    params.record_tensor_accesses = ditem.device_record_tensor_access;
    params.function_library = ditem.function_library.get();
    // Set the device_context for this node id, if it exists.
    params.op_device_context = ditem.device->deviceContextForNode(id);

    // Don't track allocations. Not implemented.
    params.track_allocations = false;


    VLOG(2) << "Process node: " << SummarizeNodeDef(node->def()) << " " << ditem.device->resourceContext();

    auto input_tensors = m_state->GetInputTensors(input_frame, input_iter);
    first_input = input_tensors + item.input_start;

    CVLOG(1, logging::kOpTracing) << "OpItem Event " << DebugString()
                                  << " event: afterDevCtx";
    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    if (tagged_node.is_dead && !IsTransferNode(node)) {
        afterCompute(true, cbs, item);
        return;
    }

    // Prepares inputs.
    bool is_input_dead = false;
    s = m_state->PrepareInputs(item, op_kernel, ditem.device, params.op_device_context, first_input, &inputs,
                               &buflocks, &input_device_contexts, &input_alloc_attrs, &is_input_dead);
    if (!s.ok()) {
        // Inspect return state for retrying on memory failure
        if (maybeMemoryFailure(s, cbs.memFailure)) {
            return;
        }
        // Clear inputs.
        m_state->ClearInputs(first_input, item.num_inputs, buflocks);
        afterRun(s, cbs);
        return;
    }

    // record input sizes
    input_size = 0;
    for (auto &inp : inputs) {
        if (inp.tensor) {
            input_size += inp->shape().num_elements();
        }
    }

    // Remember tickets for reffed inputs, they may be modified by the op
    reffedEntries.clear();
    for (auto entry = first_input; entry != first_input + item.num_inputs; ++entry) {
        if (entry->ref) {
            has_ref_input = true;
            reffedEntries.push_back(entry);
        }
    }

    // Set up compute params.
    params.op_kernel = op_kernel;
    params.frame_iter = tf::FrameAndIter(input_frame->frame_id, input_iter);
    params.is_input_dead = is_input_dead;
    params.output_attr_array = item.output_attrs();

    CVLOG(1, logging::kOpTracing) << "OpItem Event " << DebugString()
                                  << " event: afterPrepInput";

    if (kernel_is_async) {
        // Asynchronous computes.
        VLOG(2) << "Launch Async kernel";
        auto async = op_kernel->AsAsync();
        DCHECK_NOTNULL(async);

        // Ensure OpKernelContext constructor will make a new eigen GPU device if
        // necessary.
        params.eigen_gpu_device = nullptr; // Force allocation
        pctx = std::make_unique<tf::OpKernelContext>(&params, item.num_outputs);

        ditem.device->ComputeAsync(async, pctx.get(), [this, cbs = std::move(cbs), &item]() {
            VLOG(2) << "Async Kernel done: " << SummarizeNodeDef(tagged_node.node->def());
            afterCompute(false, cbs, item);
        });
    } else {
        // Synchronous computes.
        VLOG(2) << "Launch sync kernel";
        pctx = std::make_unique<tf::OpKernelContext>(&params, item.num_outputs);

        DCHECK_NOTNULL(op_kernel);
        ditem.device->Compute(op_kernel, pctx.get());

        VLOG(2) << "Kernel done: " << SummarizeNodeDef(tagged_node.node->def());
        afterCompute(false, cbs, item);
    } // if (kernel_is_async)
}

void ExecTask::afterCompute(bool is_dead, const Callbacks &cbs, const tf::remote::NodeItem &item)
{
    CVLOG(1, logging::kOpTracing) << "OpItem Event " << DebugString()
                                  << " event: afterCompute";
    // `cbs.done` should be called last as `this` would be deleted in it.
    auto &device = ditem.device;
    ExecutorState::EntryVector outputs;
    tf::Status s;

    if (is_dead) {
        outputs.resize(item.num_outputs);
        buflocks.clear();
    } else {
        // Inspect return state for retrying on memory failure
        if (maybeMemoryFailure(pctx->status(), cbs.memFailure)) {
            return;
        }

        s = m_state->ProcessOutputs(item, pctx.get(), device, &outputs);
        // clear locks, we don't need them, and they may be deleted when
        // update ref entry
        buflocks.clear();
        // Update ref entry tickets
        updateRefEntryTickets(reffedEntries);
    }

    // Clears inputs.
    m_state->ClearInputs(first_input, item.num_inputs, buflocks);

    CVLOG(1, logging::kOpTracing) << "OpItem Event " << DebugString()
                                  << " event: afterClearInput";

    // Mark completed
    auto input_frame = tagged_node.input_frame;
    const int64_t input_iter = tagged_node.input_iter;
    const int id = tagged_node.node->id();
    m_state->MaybeMarkCompleted(input_frame, input_iter, id);

    // propagate outputs
    if (s.ok()) {
        m_state->PropagateOutputs(tagged_node, &item, &outputs, &ready);
    }
    outputs.clear();

    // record tensor access
    if (s.ok() && !is_dead && ditem.device_record_tensor_access) {
        // Get the list of all tensors accessed during the execution
        tf::TensorReferenceVector accessed;
        pctx->retrieve_accessed_tensors(&accessed);
        // callee takes ownership of the vector
        device->ConsumeListOfAccessedTensors(pctx->op_device_context(), accessed);
    }

    CVLOG(1, logging::kOpTracing) << "OpItem Event " << DebugString()
                                  << " event: afterPropOut";
    // Post process
    // call node done and cbs.done
    afterRun(s, cbs);
}

void ExecTask::updateRefEntryTickets(const std::vector<Entry *> &entries)
{
    auto impl = m_state->impl_;

    for (auto &entry : entries) {
        Entry::MaybeLock l(entry);

        auto tensor = entry->ref;
        DCHECK(tensor);
        auto buf = tf::remote::PagingHelper::bufferOf(*tensor);
        DCHECK(buf);
        auto root_buf = buf->root_buffer();
        DCHECK(root_buf);

        auto perop = PerOpAllocator::downcast(buf->allocator());
        DCHECK(perop);
        auto ticket = perop->resourceContext().ticket();

        auto tree = entry->alloc_tree;
        DCHECK(tree);
        if (tree->root_buf != root_buf) {
            // The entry has changed it's buffer, remove it from old tree,
            // and any other entry references the same tensor.
            VLOG(2) << "Update allocation ticket from " << tree->ticket << " to " << ticket;

            EntryVec needUpdate;
            impl->removeFromBufferTree(entry, &needUpdate);
            // and update entries as if it's new
            for (auto e : needUpdate) {
                impl->updateBufferTree(e, ticket);
            }
        }
    }
}

void ExecTask::afterRun(const tf::Status &s, const Callbacks &cbs)
{
    DCHECK(ditem.device);
    if (s.ok()) {
        // save succeed estimation
        auto usage = estimatedUsage(ditem.device->resourceContext().spec());
        m_state->impl_->saveSucceedUsageForNode(tagged_node.node->name(), usage);
    }
    auto completed = m_state->NodeDone(s, tagged_node.node, ditem.device.get(), params.rendezvous, ready);

    num_finished_ops.notify();

    // `this` may be deleted in done
    cbs.done();

    // Do this after cbs.done, because m_state may be accessed in cbs.done
    if (completed) {
        // `m_state` may be deleted in Finish
        m_state->Finish();
    }
}

bool ExecTask::maybeMemoryFailure(const tf::Status &s, const MemFailCallback &memFailure)
{
    if (s.code() == tf::error::RESOURCE_EXHAUSTED) {
        // we didn't implement rollback. So this can only happen for non ref input ops
        // DCHECK(!has_ref_input);

        // also release locks
        buflocks.clear();

        ++failureTimes;

        DCHECK(ditem.device);
        resources::merge(failedAlloc, ditem.device->failedResourceRequest());

        if (memFailure && memFailure()) {
            VLOG(1) << "OOM happened and catched by scheduler: " << DebugString();
            return true;
        }
        VLOG(1) << "OOM happened and propagated: " << DebugString();
    }
    // This is either not a OOM error, or the scheduler is not willing to handle it,
    // just go through normal handling
    return false;
}

ResourceContext &ExecTask::resourceContext() const
{
    DCHECK(ditem.device);
    return ditem.device->resourceContext();
}

ExecTask::~ExecTask()
{
    // At this time m_state may already be deleted.
    if (owns_kernel && op_kernel) {
        DCHECK(ditem.function_library);
        deleteKernel(op_kernel, ditem.function_library.get());
    }
}

} // namespace salus::oplib::tensorflow
