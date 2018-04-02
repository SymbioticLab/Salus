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

#ifndef MULTIDEVICEEXECUTORSTATEIMPL_H
#define MULTIDEVICEEXECUTORSTATEIMPL_H

/*
 * Make sure tensorflow_headers is included first before
 * any other headers, so we can correctly override TF logging
 * with ours.
 */
#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "execution/executionengine.h"
#include "oplibraries/tensorflow/v2/md_executor.h"
#include "oplibraries/tensorflow/v2/tensorutils.h"
#include "oplibraries/tensorflow/v2/graphview.h"
#include "utils/containerutils.h"
#include "utils/threadutils.h"
#include "utils/pointerutils.h"
#include <boost/intrusive/list.hpp>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace gtl = ::tensorflow::gtl;

struct DeviceSpec;
namespace salus::oplib::tensorflow {

/*
namespace nodestats {
inline int64_t NowInUsec()
{
    return tf::Env::Default()->NowMicros();
}

void SetScheduled(tf::NodeExecStats *nt, int64_t t);

void SetAllStart(tf::NodeExecStats *nt);

void SetOpStart(tf::NodeExecStats *nt);

void SetOpEnd(tf::NodeExecStats *nt);

void SetAllEnd(tf::NodeExecStats *nt);

void SetOutput(tf::NodeExecStats *nt, int slot, const tf::Tensor *v);

void SetMemory(tf::NodeExecStats *nt, tf::OpKernelContext *ctx);

void SetReferencedTensors(tf::NodeExecStats *nt, const tf::TensorReferenceVector &tensors);

bool SetTimelineLabel(tf::NodeExecStats *node_stats, const tf::Node *node);
} // namespace nodestats
 */

using TensorValueVec = gtl::InlinedVector<tf::TensorValue, 4>;
using AllocatorAttributeVec = gtl::InlinedVector<tf::AllocatorAttributes, 4>;
using DeviceContextVec = gtl::InlinedVector<tf::DeviceContext *, 4>;
using EntryVec = gtl::InlinedVector<Entry *, 4>;
// using BufferLockVec = gtl::InlinedVector<boost::shared_lock<boost::upgrade_mutex>, 4>;
using BufferLockVec = std::vector<boost::shared_lock<boost::upgrade_mutex>>;
using BufferMutexSet = std::unordered_set<boost::upgrade_mutex *>;

class PerTaskDevice;
class ExecutorState;
class ExecutorImpl : public tf::Executor
{
public:
    ExecutorImpl(MultiDeviceExecutorParams &&p, std::unique_ptr<const tf::Graph> &&g);

    ~ExecutorImpl() override;

    tf::Status Initialize();

    void RunAsync(const Args &args, DoneCallback done) override;

private:
    friend class ExecutorState;
    friend class ExecTask;

    size_t handlePagingRequest(uint64_t oldTicket, std::unique_ptr<ResourceContext> &&rctx);
    void forceEvicted();

    /**
     * Remove entry from it's associated buffer tree.
     * If needUpdate is not nullptr, then other entries that is reference
     * to this entry is also removed, and added to needUpdate, including this
     * entry.
     */
    void removeFromBufferTree(const Entry *entry, EntryVec *needUpdate);
    void updateBufferTree(Entry *entry, uint64_t ticket);

    void saveSucceedUsageForNode(const std::string &name, const Resources &res)
    {
        sstl::Guard g(usage_mu_);
        if (!resources::contains(cachedUsages_[name], res)) {
            cachedUsages_[name] = res;
        }
    }

    std::optional<Resources> cachedUsageForNode(const std::string &name)
    {
        sstl::Guard g(usage_mu_);
        return sstl::optionalGet(cachedUsages_, name);
    }

    struct DeviceItem
    {
        std::shared_ptr<salus::oplib::tensorflow::PerTaskDevice> device = nullptr;
        std::shared_ptr<tf::FunctionLibraryRuntime> function_library = nullptr;
        bool device_record_tensor_access = false;
    };

    tf::Status LookupDevice(const DeviceSpec &spec, std::unique_ptr<ResourceContext> &&rctx, DeviceItem *item);

    tf::Status LookupTFDevice(const DeviceSpec &spec, tf::Device **tfdev);

    struct ControlFlowInfo
    {
        gtl::FlatSet<std::string> unique_frame_names;
        std::vector<std::string> frame_names;
    };

    struct FrameInfo
    {
        FrameInfo() = default;
        // The total number of inputs to a frame.
        int input_count = 0;

        // The total number of input tensors of a frame.
        // == sum(nodes[*].num_inputs()) where nodes are the nodes in the frame.
        int total_inputs = 0;

        // Used to determine the next place to allocate space in the
        // pending_counts data structure we'll eventually construct
        tf::PendingCounts::Layout pending_counts_layout;

        // Each frame has its own PendingCounts only for the nodes in the frame.
        sstl::owner<tf::PendingCounts *> pending_counts = nullptr; // Owned

        // The nodes in a frame. Used only for debugging.
        sstl::owner<std::vector<const tf::Node *> *> nodes = nullptr; // Owned

        ~FrameInfo()
        {
            delete pending_counts;
            delete nodes;
        }
    };

    static tf::Status BuildControlFlowInfo(sstl::not_null<const tf::Graph *> graph, ControlFlowInfo *cf_info);
    void InitializePending(sstl::not_null<const tf::Graph *> graph, const ControlFlowInfo &cf_info);

    FrameInfo *EnsureFrameInfo(const std::string &fname)
    {
        auto slot = &frame_info_[fname];
        if (*slot == nullptr) {
            *slot = new FrameInfo;
        }
        return *slot;
    }

    // Instantiate the op kernel for node.
    POpKernel SetupKernel(sstl::not_null<const tf::Node *> node, const DeviceItem &ditem);

    std::mutex kernel_dev_mu_;
    std::unordered_map<std::string, const tf::Device*> kernel_dev_ GUARDED_BY(kernel_dev_mu_);

    MultiDeviceExecutorParams params_;
    std::unique_ptr<const tf::Graph> graph_;
    GraphView gview_;
    ExecutionContext inserter_;

    // Active entries. Used for handle paging request
    std::mutex entry_mu_;
    boost::intrusive::list<TensorBufferTree, boost::intrusive::constant_time_size<false>> buffer_trees_;
    std::unordered_multimap<uint64_t, TensorBufferTree *> active_buffers_;
    std::unordered_set<ExecutorState *> active_states_;

    // Root nodes (with no in edges) that should form the initial ready queue
    std::vector<const tf::Node *> root_nodes_;

    // Client terminated recv nodes for inputs, this should also in root nodes.
    std::unordered_set<const tf::Node *> client_recv_nodes_;

    // Mapping from frame name to static information about the frame.
    // TODO(yuanbyu): We could cache it along with the graph so to avoid
    // the overhead of constructing it for each executor instance.
    gtl::FlatMap<std::string, FrameInfo *> frame_info_;

    // Known succeed node resource usage
    std::unordered_map<std::string, Resources> cachedUsages_;
    std::mutex usage_mu_;

    TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
};

// The state associated with one invocation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
struct DeviceItem;
class ExecTask;
class PerTaskDevice;
class ExecutorState
{
public:
    ExecutorState(const tf::Executor::Args &args, ExecutorImpl *impl);
    ~ExecutorState();

    void RunAsync(const tf::Executor::DoneCallback &done);

    // Something wrong happened.
    void ForceInterrupt(const tf::Status &s);

private:
    friend class ExecTask;

    struct TaggedNode;
    using TaggedNodeSeq = gtl::InlinedVector<TaggedNode, 8>;
    using EntryVector = gtl::InlinedVector<Entry, 4>;

    struct IterationState
    {
        explicit IterationState(const tf::PendingCounts *pending_counts, int total_input_tensors)
            : input_tensors(new Entry[total_input_tensors])
            , total_input_tensors(total_input_tensors)
            , outstanding_ops(0)
            , outstanding_frame_count(0)
            , counts_(*pending_counts)
        { // Initialize with copy of *pending_counts
        }

        // The state of an iteration.

        // One copy per iteration. For iteration k, i-th node's j-th input is in
        // input_tensors[k][impl_->nodes[i].input_start + j]. An entry is either
        // a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
        //
        // NOTE: No need to protect input_tensors[i] by any locks because it
        // is resized once. Each element of tensors_ is written once by the
        // source node of an edge and is cleared by the destination of the same
        // edge. The latter node is never run concurrently with the former node.
        Entry *input_tensors;
        int total_input_tensors;

        // The number of outstanding ops for each iteration.
        int outstanding_ops;

        // The number of outstanding frames for each iteration.
        int outstanding_frame_count;
        int pending(tf::PendingCounts::Handle h)
        {
            return counts_.pending(h);
        }
        int decrement_pending(tf::PendingCounts::Handle h, int v)
        {
            return counts_.decrement_pending(h, v);
        }
        // Mark a merge node as live
        // REQUIRES: Node corresponding to "h" is a merge node
        void mark_live(tf::PendingCounts::Handle h)
        {
            counts_.mark_live(h);
        }
        // Mark a node to show that processing has started.
        void mark_started(tf::PendingCounts::Handle h)
        {
            counts_.mark_started(h);
        }
        // Mark a node to show that processing has completed.
        void mark_completed(tf::PendingCounts::Handle h)
        {
            counts_.mark_completed(h);
        }
        tf::PendingCounts::NodeState node_state(tf::PendingCounts::Handle h)
        {
            return counts_.node_state(h);
        }

        int dead_count(tf::PendingCounts::Handle h)
        {
            return counts_.dead_count(h);
        }
        void increment_dead_count(tf::PendingCounts::Handle h)
        {
            counts_.increment_dead_count(h);
        }
        void adjust_for_activation(tf::PendingCounts::Handle h, bool increment_dead, int *pending_result,
                                   int *dead_result)
        {
            counts_.adjust_for_activation(h, increment_dead, pending_result, dead_result);
        }

        ~IterationState()
        {
            delete[] input_tensors;
        }

    private:
        tf::PendingCounts counts_;
    };

    struct FrameState
    {
        explicit FrameState(ExecutorImpl *impl, int parallel_iters)
            : executor(impl)
            , max_parallel_iterations(parallel_iters)
            , num_outstanding_iterations(1)
        {
        }

        // A new frame is created for each loop. Execution starts at iteration 0.
        // When a value at iteration 0 passes through a NextIteration node,
        // iteration 1 is created and starts running. Note that iteration 0 may
        // still be running so multiple iterations may run in parallel. The
        // frame maintains the state of iterations in several data structures
        // such as pending_count and input_tensors. When iteration 0 completes,
        // we garbage collect the state of iteration 0.
        //
        // A frame instance is considered "done" and can be garbage collected
        // if all its inputs have entered and all its iterations are "done".
        //
        // A frame manages the live iterations of an iterative computation.
        // Iteration i is considered "done" when there are no outstanding ops,
        // frames at iteration i are done, all recvs for this iteration are
        // completed, and iteration i-1 is done. For iteration 0, we instead
        // wait for there to be no more pending inputs of the frame.
        //
        // Frames and iterations are garbage collected once they are done.
        // The state we need to keep around is highly dependent on the
        // parallelism enabled by the scheduler. We may want to have the
        // scheduler dynamically control the outstanding number of live
        // parallel frames and iterations. To reduce the state space, the
        // scheduler might want to schedule ops in inner frames first and
        // lower iterations first.
        //
        // This frame state is mostly initialized lazily on demand so we
        // don't introduce unnecessary overhead.

        // The executor the frame is in.
        ExecutorImpl *executor = nullptr;

        // The name of this frame, which is the concatenation of its parent
        // frame name, the iteration of the parent frame when this frame was
        // created, and the value of the attr 'frame_name'.
        std::string frame_name;

        // The unique id for this frame. Generated by fingerprinting
        // frame_name.
        uint64_t frame_id = -1;

        // The iteration id of its parent frame when this frame is created.
        // -1 if there is no parent frame. The frame_name/parent_iter pair
        // uniquely identifies this FrameState.
        int64_t parent_iter = -1;

        // The FrameState of its parent frame.
        FrameState *parent_frame = nullptr;

        // The maximum allowed number of parallel iterations.
        const int max_parallel_iterations;

        // The number of inputs this frame is still waiting.
        int num_pending_inputs = 0;

        // The highest iteration number we have reached so far in this frame.
        int64_t iteration_count GUARDED_BY(mu) = 0;

        // The number of outstanding iterations.
        int num_outstanding_iterations GUARDED_BY(mu) = 1;

        // The active iteration states of this frame.
        gtl::InlinedVector<sstl::owner<IterationState *>, 12> iterations;

        // The NextIteration nodes to enter a new iteration. If the number of
        // outstanding iterations reaches the limit, we will defer the start of
        // the next iteration until the number of outstanding iterations falls
        // below the limit.
        std::vector<std::pair<const tf::Node *, Entry>> next_iter_roots GUARDED_BY(mu);

        // The values of the loop invariants for this loop. They are added into
        // this list as they "enter" the frame. When a loop invariant enters,
        // we make it available to all active iterations. When the frame starts
        // a new iteration, we make all the current loop invariants available
        // to the new iteration.
        std::vector<std::pair<const tf::Node *, Entry>> inv_values GUARDED_BY(mu);

        // The list of dead exit nodes for the current highest iteration. We
        // will only "execute" the dead exits of the final iteration.
        std::vector<const tf::Node *> dead_exits GUARDED_BY(mu);

        // Static information specific to this frame.
        tf::PendingCounts *pending_counts = nullptr;
        int total_input_tensors = 0;
        std::vector<const tf::Node *> *nodes = nullptr;

        // Lock ordering: ExecutorState.mu_ < mu.
        std::mutex mu;

        void InitializeFrameInfo(const tf::string &enter_name)
        {
            auto it_frame_info = executor->frame_info_.find(enter_name);
            DCHECK(it_frame_info != executor->frame_info_.end());
            ExecutorImpl::FrameInfo *finfo = it_frame_info->second;
            pending_counts = finfo->pending_counts;
            total_input_tensors = finfo->total_inputs;
            num_pending_inputs = finfo->input_count;
            nodes = finfo->nodes;
        }

        inline IterationState *GetIteration(int64_t iter) EXCLUSIVE_LOCKS_REQUIRED(mu)
        {
            int index = iter % iterations.size();
            return iterations[index];
        }

        inline void SetIteration(int64_t iter, IterationState *state) EXCLUSIVE_LOCKS_REQUIRED(mu)
        {
            int index = iter % iterations.size();
            DCHECK(state == nullptr || iterations[index] == nullptr);
            iterations[index] = state;
        }

        // Decrement the outstanding op count and clean up the iterations in the
        // frame. Return true iff the execution of the frame is done.
        inline bool DecrementOutstandingOps(const GraphView &gview, int64_t iter,
                                            TaggedNodeSeq *ready)
        {
            sstl::Guard l(mu);
            return DecrementOutstandingOpsLocked(gview, iter, ready);
        }

        // Decrement the outstanding op count and clean up the iterations in the
        // frame. Return true iff the execution of the frame is done.
        inline bool DecrementOutstandingOpsLocked(const GraphView &gview, int64_t iter,
                                                  TaggedNodeSeq *ready) EXCLUSIVE_LOCKS_REQUIRED(mu)
        {
            IterationState *istate = GetIteration(iter);
            istate->outstanding_ops--;
            if (istate->outstanding_ops != 0) {
                return false;
            } else {
                return CleanupIterations(gview, iter, ready);
            }
        }

        // Returns true if the computation in the frame is completed.
        inline bool IsFrameDone() EXCLUSIVE_LOCKS_REQUIRED(mu)
        {
            return (num_pending_inputs == 0 && num_outstanding_iterations == 0);
        }

        // Returns true if the iteration of the frame is completed.
        bool IsIterationDone(int64_t iter) EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Increments the iteration id. If this is a new iteration, initialize it.
        void IncrementIteration(const GraphView &gview, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Activate all the deferred NextIteration nodes in a new iteration.
        void ActivateNexts(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Activate all the current loop invariants in a new iteration.
        void ActivateLoopInvs(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Add a new loop invariant and make it available to all active iterations.
        void AddLoopInv(const NodeItem &item, const Entry &value, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Activate the successors of a node. Contents of *outputs are left in an
        // indeterminate state after returning from this method.
        void ActivateNodes(const NodeItem &item, bool is_dead, int64_t iter,
                           sstl::not_null<EntryVector *> outputs, TaggedNodeSeq *ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Cleanup iterations of this frame starting from iteration iter.
        bool CleanupIterations(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        ~FrameState()
        {
            for (auto &iteration : iterations) {
                delete iteration;
                iteration = nullptr;
            }
        }
    };

    // A tagged node: <frame*, iter, node*>.
    struct TaggedNode
    {
        const tf::Node *node = nullptr;
        FrameState *input_frame = nullptr;
        int64_t input_iter = -1;
        bool is_dead = false;

        TaggedNode(const tf::Node *t_node, FrameState *in_frame, int64_t in_iter, bool dead)
        {
            node = t_node;
            input_frame = in_frame;
            input_iter = in_iter;
            is_dead = dead;
        }
    };

    const bool vlog_; // true if VLOG_IS_ON(1). Used to check vlog cheaply.
    bool forceInterrupted = false;

    tf::ShapeRefiner refiner_;
    std::unordered_map<std::string, tf::PartialTensorShape> sendShapes_;
    std::mutex refinerMu_;

    void addNodeToRefiner(const TaggedNode &tn);

    void fetchRecvShape(const tf::Node *n);

    inline auto shapeForNode(const tf::Node *n)
    {
        sstl::Guard l(refinerMu_);
        return refiner_.GetContext(n);
    }

    int64_t step_id_;
    // Not owned.
    tf::Rendezvous *rendezvous_;
    tf::SessionState *session_state_;
    tf::TensorStore *tensor_store_;
    // Step-local container.
    tf::ScopedStepContainer *step_container_;
    tf::StepStatsCollector *stats_collector_;
    tf::CallFrameInterface *call_frame_;
    ExecutorImpl *impl_;
    tf::CancellationManager *cancellation_manager_;
    tf::Executor::Args::Runner runner_;

    bool sync_on_finish_;

    // Owned.

    // A flag that is set on error after the frame state has been
    // dumped for diagnostic purposes.
    bool dumped_on_error_ = false;

    // The root frame in which the execution of this step is started.
    FrameState *root_frame_;

    // Invoked when the execution finishes.
    tf::Executor::DoneCallback done_cb_;

    std::atomic_int_fast32_t num_outstanding_ops_;
    std::atomic_int_fast32_t num_emitted_ops_;
    sstl::semaphore num_finished_ops_;

    std::mutex mu_;
    tf::Status status_ GUARDED_BY(mu_);

    // Mapping from frame name to outstanding frames. A new frame is created
    // at some iteration of an active frame. So the unique key for the new
    // child frame is composed of the name of the parent frame, the iteration
    // number at which the parent frame is creating the new frame, and the
    // name of the new frame from nodedef.
    gtl::FlatMap<std::string, FrameState *> outstanding_frames_ GUARDED_BY(mu_);

    // The unique name of a frame.
    inline std::string MakeFrameName(FrameState *frame, int64_t iter_id, std::string name)
    {
        return tf::strings::StrCat(frame->frame_name, ";", iter_id, ";", name);
    }

    // Find an existing or create a new child frame in the frame 'frame' at
    // iteration 'iter'.
    void FindOrCreateChildFrame(FrameState *frame, int64_t iter, const tf::Node *node, FrameState **child);

    // Delete a frame. Called when the frame is done.
    void DeleteFrame(FrameState *frame, TaggedNodeSeq *ready);

    // Cleanup frames and iterations starting from frame/iter. Called when
    // a child frame is done.
    void CleanupFramesIterations(FrameState *frame, int64_t iter, TaggedNodeSeq *ready);

    // Process a ready node and submit to execution engine in current thread.
    void Process(TaggedNode node);

    // Before invoking item->kernel, fills in its "inputs".
    tf::Status PrepareInputs(const NodeItem &item, sstl::not_null<tf::OpKernel *> kernel,
                             const std::shared_ptr<PerTaskDevice> &device,
                             tf::DeviceContext *device_context, Entry *first_input, TensorValueVec *inputs,
                             sstl::not_null<BufferLockVec *> buflocks, DeviceContextVec *input_device_contexts,
                             AllocatorAttributeVec *input_alloc_attrs, bool *is_input_dead);

    // After item->kernel computation is done, processes its outputs.
    tf::Status ProcessOutputs(const NodeItem &item, tf::OpKernelContext *ctx,
                              const std::shared_ptr<PerTaskDevice> &device, EntryVector *outputs);

    // After item->kernel computation is done, clear its inputs.
    void ClearInputs(Entry *first, size_t num, BufferLockVec &buflocks);

    // After processing the outputs, propagates the outputs to their dsts.
    // Contents of *outputs are left in an indeterminate state after
    // returning from this method.
    void PropagateOutputs(const TaggedNode &tagged_node, const NodeItem &item,
                          sstl::not_null<EntryVector *> outputs, TaggedNodeSeq *ready);

    // "node" just finishes. Takes ownership of "stats". Returns true if
    // execution has completed.
    bool NodeDone(const tf::Status &s, const tf::Node *node, const tf::Device *device,
                  tf::Rendezvous *rendezvous, const TaggedNodeSeq &ready);

    // Schedule all the expensive nodes in 'ready'
    void ScheduleReady(const TaggedNodeSeq &ready);

    // For debugging/logging only.
    inline void MaybeMarkCompleted(FrameState *frame, int64_t iter, int64_t node_id)
    {
        // TODO(misard) Replace with a finer-grain enabling flag once we
        // add better optional debugging support.
        if (vlog_ && VLOG_IS_ON(1)) {
            const auto *item = impl_->gview_.node(node_id);
            sstl::Guard l(frame->mu);
            frame->GetIteration(iter)->mark_completed(item->pending_id);
        }
    }

    // Provide debugging output about an outstanding node in the executor.
    void DumpPendingNodeState(const int node_id, const Entry *input_vector,
                              bool show_nodes_with_no_ready_inputs);
    void DumpActiveNodeState(const int node_id, const Entry *input_vector);

    // Provide debugging output about an outstanding iteration in the executor.
    void DumpIterationState(const FrameState *frame, IterationState *iteration);

    // Provide debugging output of the state of the executor.
    void DumpState();
    const tf::Tensor *GetTensorValueForDump(const Entry &input);

    // Clean up when this executor is done.
    void Finish();

    // A standalone routine for this expression so that we can express
    // that we don't want thread safety analysis on this reference (it's
    // safe to do without the lock because the iterations array never
    // resizes and this particular iteration's array element will not
    // be changed out from under us because the iteration is still alive).
    Entry *GetInputTensors(FrameState *input_frame, int64_t input_iter) const NO_THREAD_SAFETY_ANALYSIS
    {
        return input_frame->GetIteration(input_iter)->input_tensors;
    }
};

} // namespace salus::oplib::tensorflow

#endif // MULTIDEVICEEXECUTORSTATEIMPL_H
