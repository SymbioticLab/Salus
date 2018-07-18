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
#include "oplibraries/tensorflow/v2/exectask.h"
#include "execution/devices.h"
#include "execution/engine/resourcecontext.h"
#include "execution/engine/iterationcontext.h"
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "oplibraries/tensorflow/v2/tfallocator.h"
#include "oplibraries/tensorflow/worker/devicecontextwithdevice.h"
#include "oplibraries/tensorflow/worker/rendezvouswithhook.h"
#include "utils/date.h"
#include "utils/macros.h"
#include "utils/cpp17.h"
#include "utils/threadutils.h"
#include "utils/debugging.h"
#include <boost/range/algorithm.hpp>
#include <sstream>

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
using FpMS = std::chrono::duration<double, std::chrono::milliseconds::period>;
using namespace std::chrono_literals;
using namespace date;

namespace salus::oplib::tensorflow {

ExecTask::ExecTask(ExecutorState *state, sstl::semaphore &num_finished_ops,
                   const ExecutorState::TaggedNode &node, const tf::OpKernelContext::Params &initial_params,
                   tf::Rendezvous *rendez, const int maxFailures)
    : m_state(state)
    , item(*m_state->impl_->gview_.node(node.node->id()))
    , rendez(rendez)
    , num_finished_ops(num_finished_ops)
    , failureTimes(0)
    , maxFailures(maxFailures)
    , tagged_node(node)
    , op_kernel(nullptr, skip_delete_opkernel)
    , kernel_is_async(false)
    , has_ref_input(false)
    , params(initial_params)
{
    params.inputs = &inputs;
    params.input_device_contexts = &input_device_contexts;
    params.input_alloc_attrs = &input_alloc_attrs;

    // pre compute estimated usage
    for (auto t : item.supported_devices) {
        calcUsageFromShape(DeviceSpec{t});
    }

    // pre compute debug string
    std::ostringstream oss;
    oss << "ExecTask(name=" << tagged_node.node->name() << ", type=" << tagged_node.node->op_def().name()
        << ", session=" << m_state->impl_->params_.session
        << ", graphHandle=" << m_state->impl_->params_.graphHandle
        << ", step_id=" << m_state->step_id_ << ")";
    m_cachedDebugString = oss.str();
}

ExecTask::DeviceTypes ExecTask::supportedDeviceTypes() const
{
    return item.supported_devices;
}

bool ExecTask::prepare(std::unique_ptr<ResourceContext> &&rctx) noexcept
{
    sstl::TimeoutWarning tw(2ms, [&](auto limit, auto dur){
        LOG(WARNING) << "ExecTask::prepare took more than " << duration_cast<FpMS>(limit)
                     << " to finish: " << duration_cast<FpMS>(dur)
                     << " for " << *this;
    });

    try {
        statusInPrepare = tf::Status::OK();

        auto &spec = rctx->spec();

        if (boost::range::count(supportedDeviceTypes(), spec.type) == 0) {
            LOG(ERROR) << "Try to prepare ExecTask on unsupported device type " << spec << " :" << *this;
            LOG(ERROR) << "Supported device types are: ";
            for (const auto &dt : supportedDeviceTypes()) {
                LOG(ERROR) << "  " << enumToString(dt);
            }
            return false;
        }

        auto s = m_state->impl_->LookupDevice(spec, std::move(rctx), &ditem);
        if (!s.ok()) {
            statusInPrepare.Update(s);
            return false;
        }

        LogAlloc() << "Pre allocated " << ditem.device->resourceContext() << " for " << *this;

    #if defined(SALUS_ENABLE_MULTI_DEVICE)
        // In case anything go wrong, don't leak resource
        sstl::ScopeGuards onReturn([this]() {
            // Release the resource context and the device we've given
            ditem.device.reset();
        });
        // Instantiate kernel if not already done
        try {
            op_kernel = m_state->impl_->SetupKernel(tagged_node.node, ditem);
        } catch (const TFException &ex) {
            if (tf::errors::IsAlreadyExists(ex.code())) {
                // found a kernel on another device
                // ignore it as we probably have another chance to prepare on another device.
                return false;
            }
            // rethrow otherwise
            throw;
        }
        kernel_is_async = (op_kernel->AsAsync() != nullptr);

        // Now we are sure things succeeded, cancel the rollback
        onReturn.dismiss();
    #endif // SALUS_ENABLE_MULTI_DEVICE

        return true;
    } catch (const TFException &ex) {
        LOG(ERROR) << "Exception caught when preparing opItem " << *this << ": " << ex.what();
        statusInPrepare = ex.code();
        return false;
    }
}

bool ExecTask::isAsync() const
{
#if defined(SALUS_ENABLE_MULTI_DEVICE)
    return kernel_is_async;
#else
    return item.kernel_is_async;
#endif // SALUS_ENABLE_MULTI_DEVICE
}

bool ExecTask::hasExactEstimation(const DeviceSpec &dev)
{
    auto usage = m_state->impl_->cachedUsageForNode(item, dev.type);
    return bool(usage);
}

Resources ExecTask::estimatedUsage(const DeviceSpec &dev)
{
    sstl::TimeoutWarning tw(1ms, [&](auto limit, auto dur){
        LOG(WARNING) << "ExecTask::estimatedUsage took more than " << duration_cast<FpMS>(limit)
                     << " to finish: " << duration_cast<FpMS>(dur)
                     << " for " << *this;
    });
    const static constexpr auto DevCapacity = 14ll * 1024ll * 1024ll * 1024ll;
    // First see if we have usage for this node in session
    // but only if we haven't failed before, otherwise,
    // the session cached usage maybe be just a lucky case
    auto usage = m_state->impl_->cachedUsageForNode(item, dev.type);
    if (usage && failureTimes == 0) {
        if (sstl::optionalGet(usage, resources::GPU0Memory) > DevCapacity) {
            LOG(WARNING) << "Cap resource usage estimation larger than device capacity: " << *usage << " capacity: " << DevCapacity;
            usage.value()[resources::GPU0Memory] = DevCapacity;
        }
        cachedUsage[dev] = *usage;
        return std::move(*usage);
    }

    // Short-cut if this task has failed before
    if (failureTimes > 0) {
        if (!failedAlloc.empty()) {
            // we don't care if operator[] inserts a new element or not.
            // it it does insert a new element, this means no estimation is available
            // anyway.
            auto usage = cachedUsage[dev];
            resources::merge(usage, failedAlloc);
            if (sstl::optionalGet(usage, resources::GPU0Memory) > DevCapacity) {
                LOG(WARNING) << "Cap resource usage estimation larger than device capacity: " << usage << " capacity: " << DevCapacity;
                usage[resources::GPU0Memory] = DevCapacity;
            }
            return usage;
        }
    }

    // Fast path from cache
    auto it = cachedUsage.find(dev);
    if (it == cachedUsage.end()) {
        return calcUsageFromShape(dev);
    }
    if (sstl::optionalGet(it->second, resources::GPU0Memory) > DevCapacity) {
        LOG(WARNING) << "Cap resource usage estimation larger than device capacity: " << it->second << " capacity: " << DevCapacity;
        it->second[resources::GPU0Memory] = DevCapacity;
    }
    return it->second;
}

Resources ExecTask::calcUsageFromShape(const DeviceSpec &dev)
{
    // Slow path to calculate the usage
    auto &res = cachedUsage[dev];

#if defined(SALUS_ENABLE_REFINER)
    const auto *node = item.node;
    if (auto ctx = m_state->shapeForNode(node)) {
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
            VLOG(3) << "    dtype " << dtype;
            double subtotal = count * tf::DataTypeSize(dtype);

            if (item.output_attrs()[i].on_host()) {
                res[cpuTag] += subtotal;
            } else {
                res[devTag] += subtotal;
            }
        }
    } else {
        LOG(WARNING) << "Shape information not available for node: " << node->name();
        res = estimateMemoryUsageForNode(item, dev);
    }
#else
    res = estimateMemoryUsageForNode(item, dev);
#endif // SALUS_ENABLE_REFINER

#if !defined(SALUS_ENABLE_STATIC_STREAM)
    // TODO: use an algorithm to decide streams
    if (dev.type == DeviceType::GPU && !isAsync()) {
        res[{ResourceType::GPU_STREAM, dev}] = 1;
    }
#endif

    return res;
}

std::string ExecTask::DebugString() const
{
    return m_cachedDebugString;
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

void ExecTask::run(Callbacks cbs) noexcept
{
    const auto &gview = m_state->impl_->gview_;
    auto *node = tagged_node.node;
    const auto input_frame = tagged_node.input_frame;
    const auto input_iter = tagged_node.input_iter;
    const auto id = node->id();

    // clear early
    params.rendezvous = nullptr;

#if defined(SALUS_ENABLE_MULTI_DEVICE)
    auto kernel = op_kernel.get();
#else
    auto kernel = item.kernel.get();
#endif // SALUS_ENABLE_MULTI_DEVICE

    try {
        if (!statusInPrepare.ok()) {
            m_state->MaybeMarkCompleted(input_frame, input_iter, id);
            afterRun(statusInPrepare, cbs);
            return;
        }

        DCHECK(kernel);
        DCHECK(ditem.device);

        // Start run
        auto s = gview.SetAllocAttrForNode(node, ditem.device.get(), kernel);
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
        params.op_device_context = ditem.device->deviceContextForNode(id, isAsync());

        // Don't track allocations. Not implemented.
        params.track_allocations = false;

        VLOG(2) << "Process node: " << node->def() << " " << ditem.device->resourceContext();

        auto input_tensors = m_state->GetInputTensors(input_frame, input_iter);
        first_input = input_tensors + item.input_start;

        LogOpTracing() << "OpItem Event " << *this << " event: afterDevCtx";
        // Only execute this node if it is not dead or it is a send/recv
        // transfer node. For transfer nodes, we need to propagate the "dead"
        // bit even when the node is dead.
        if (tagged_node.is_dead && !IsTransferNode(node)) {
            afterCompute(true, cbs);
            return;
        }

        // Prepares inputs.
        bool is_input_dead = false;
        s = m_state->PrepareInputs(item, kernel, ditem.device, params.op_device_context, first_input,
                                   &inputs, &buflocks, &input_device_contexts, &input_alloc_attrs,
                                   &is_input_dead);
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
        params.op_kernel = kernel;
        params.frame_iter = tf::FrameAndIter(input_frame->frame_id, input_iter);
        params.is_input_dead = is_input_dead;
        params.output_attr_array = item.output_attrs();

        LogOpTracing() << "OpItem Event " << *this << " event: afterPrepInput";

        if (isAsync()) {
            // Asynchronous computes.
            VLOG(2) << "Launch Async kernel";
            auto async = kernel->AsAsync();
            DCHECK_NOTNULL(async);

            // Ensure OpKernelContext constructor will make a new eigen GPU device if
            // necessary.
            params.eigen_gpu_device = nullptr; // Force allocation
            pctx = std::make_unique<tf::OpKernelContext>(&params, item.num_outputs);

            ditem.device->ComputeAsync(async, pctx.get(), [this, cbs = std::move(cbs)]() {
                VLOG(2) << "Async Kernel done: " << tagged_node.node->def();
                afterCompute(false, cbs);
            });
        } else {
            // Synchronous computes.
            VLOG(2) << "Launch sync kernel";
            pctx = std::make_unique<tf::OpKernelContext>(&params, item.num_outputs);

            DCHECK_NOTNULL(kernel);
            ditem.device->Compute(kernel, pctx.get());

            VLOG(2) << "Kernel done: " << tagged_node.node->def();
            afterCompute(false, cbs);
        } // if (kernel_is_async)
    } catch (const TFException &ex) {
        LOG(ERROR) << "Exception caught when preparing opItem " << *this << ": " << ex.what();
        afterRun(ex.code(), cbs);
    }
}

void ExecTask::afterCompute(bool is_dead, const Callbacks &cbs)
{
    LogOpTracing() << "OpItem Event " << *this << " event: afterCompute";
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

    LogOpTracing() << "OpItem Event " << *this << " event: afterClearInput";

    // Mark completed
    auto input_frame = tagged_node.input_frame;
    const int64_t input_iter = tagged_node.input_iter;
    const int id = tagged_node.node->id();
    m_state->MaybeMarkCompleted(input_frame, input_iter, id);

    // propagate outputs
    if (s.ok()) {
        m_state->PropagateOutputs(tagged_node, item, &outputs, &ready);
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

    LogOpTracing() << "OpItem Event " << *this << " event: afterPropOut";
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
        if (!buf) {
            // shouldn't happen very often
            LOG(WARNING) << "Skip one entry with nullptr buf in updateRefEntryTickets. Probably OOM caused an empty tensor";
            continue;
        }
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

            PEntryVec needUpdate;
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
        // save succeed peak resource usage as an estimation
        m_state->impl_->saveSucceedUsageForNode(item, resourceContext().spec().type,
                                                ditem.device->peakResourceUsage());
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
    if (tf::errors::IsResourceExhausted(s)) {
        // we didn't implement rollback. So this can only happen for non ref input ops
        // DCHECK(!has_ref_input);

        // also release locks
        buflocks.clear();

        ++failureTimes;

        DCHECK(ditem.device);
        resources::merge(failedAlloc, ditem.device->failedResourceRequest());
        resources::removeInvalid(failedAlloc);

        if (failureTimes == 11) {
            auto eu = estimatedUsage(devices::GPU0);
            LOG(WARNING) << "Failed more than 10 times: " << failureTimes
                         << " current estimation:" << eu
                         << " total failed request: " << failedAlloc << " " << DebugString();
        }

        if (memFailure && memFailure()) {
            VLOG(1) << "OOM happened and caught by scheduler: " << *this;
            return true;
        }
        VLOG(1) << "OOM happened and propagated: " << *this;
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

uint64_t ExecTask::graphId() const
{
    return m_state->ictx_->graphId();
}

ExecTask::~ExecTask() = default;

} // namespace salus::oplib::tensorflow
