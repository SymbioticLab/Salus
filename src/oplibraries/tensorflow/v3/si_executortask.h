//
// Created by peifeng on 7/19/18.
//

#ifndef SALUS_OPLIB_TENSORFLOW_SIMPLE_ITER_EXECUTORTASK_H
#define SALUS_OPLIB_TENSORFLOW_SIMPLE_ITER_EXECUTORTASK_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "execution/iterationtask.h"
#include "oplibraries/tensorflow/v2/graphview.h"
#include "oplibraries/tensorflow/v2/tensorutils.h"
#include "oplibraries/tensorflow/v3/si_executor_impl.h"

namespace salus::oplib::tensorflow {

using TensorValueVec = gtl::InlinedVector<tf::TensorValue, 4>;
using AllocatorAttributeVec = gtl::InlinedVector<tf::AllocatorAttributes, 4>;
using DeviceContextVec = gtl::InlinedVector<tf::DeviceContext *, 4>;
using PEntryVec = gtl::InlinedVector<Entry *, 4>;
using EntryVec = gtl::InlinedVector<Entry, 4>;

class SIExecutorState;
class SIExecutorTask : public IterationTask
{
    SIExecutor &m_impl;
    tf::CancellationManager &m_cm;

    std::unique_ptr<SIExecutorState> m_state;

public:
    SIExecutorTask(SIExecutor &impl, const tf::Executor::Args &args, tf::Executor::DoneCallback done);

    uint64_t graphId() const override
    {
        return m_impl.graph_id_;
    }

    bool prepare() override;

    ResStats estimatedPeakAllocation(const DeviceSpec &) const override
    {
        return {};
    }

    void runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept override;

    void cancel() override
    {
        m_cm.StartCancel();
    }

    bool isCanceled() const override
    {
        return m_cm.IsCancelled();
    }

    bool isExpensive() const override
    {
        return m_impl.is_main_iter;
    }
};

class SIExecutorState
{
public:
    SIExecutorState(SIExecutor &impl, const tf::Executor::Args &args, tf::Executor::DoneCallback done);

    ~SIExecutorState();

    void runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept;

private:
    // Contains a value for [node->id()] for the device context assigned by the
    // device at the beginning of a step.
    tf::DeviceContextMap device_context_map_;

    struct TaggedNode;
    using TaggedNodeSeq = gtl::InlinedVector<TaggedNode, 8>;

    struct IterationState
    {
        explicit IterationState(const tf::PendingCounts &pending_counts, int total_input_tensors)
            : input_tensors(static_cast<size_t>(total_input_tensors))
            , total_input_tensors(total_input_tensors)
            , outstanding_ops(0)
            , outstanding_frame_count(0)
            , counts_(pending_counts) // Initialize with copy of pending_counts
        {
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
        std::vector<Entry> input_tensors;
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

    private:
        tf::PendingCounts counts_;
    };

    struct FrameState
    {
        explicit FrameState(SIExecutor &impl, int parallel_iters)
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
        SIExecutor &executor;

        // The name of this frame, which is the concatenation of its parent
        // frame name, the iteration of the parent frame when this frame was
        // created, and the value of the attr 'frame_name'.
        std::string frame_name;

        // The unique id for this frame. Generated by fingerprinting
        // frame_name.
        uint64_t frame_id = 0;

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
        gtl::InlinedVector<std::unique_ptr<IterationState>, 12> iterations;

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

        // Lock ordering: SIExecutorState.mu_ < mu.
        std::mutex mu;

        void InitializeFrameInfo(const std::string &enter_name)
        {
            auto it = executor.frame_info_.find(enter_name);
            DCHECK(it != executor.frame_info_.end());
            auto &finfo = it->second;
            pending_counts = finfo.pending_counts.get();
            total_input_tensors = finfo.total_inputs;
            num_pending_inputs = finfo.input_count;
            nodes = &finfo.nodes;
        }

        inline IterationState *GetIteration(int64_t iter) EXCLUSIVE_LOCKS_REQUIRED(mu)
        {
            auto index = iter % iterations.size();
            return iterations[index].get();
        }

        inline void SetIteration(int64_t iter, std::unique_ptr<IterationState> &&state)
            EXCLUSIVE_LOCKS_REQUIRED(mu)
        {
            auto index = iter % iterations.size();
            DCHECK(state == nullptr || iterations[index] == nullptr);
            iterations[index] = std::move(state);
        }

        // Decrement the outstanding op count and clean up the iterations in the
        // frame. Return true iff the execution of the frame is done.
        inline bool DecrementOutstandingOps(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
        {
            auto l = sstl::with_guard(mu);
            return DecrementOutstandingOpsLocked(gview, iter, ready);
        }

        // Decrement the outstanding op count and clean up the iterations in the
        // frame. Return true iff the execution of the frame is done.
        inline bool DecrementOutstandingOpsLocked(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu)
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
        void IncrementIteration(const GraphView &gview, TaggedNodeSeq *ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Activate all the deferred NextIteration nodes in a new iteration.
        void ActivateNexts(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Activate all the current loop invariants in a new iteration.
        void ActivateLoopInvs(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Add a new loop invariant and make it available to all active iterations.
        void AddLoopInv(const NodeItem &item, const Entry &entry, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Activate the successors of a node. Contents of *outputs are left in an
        // indeterminate state after returning from this method.
        void ActivateNodes(const NodeItem &item, bool is_dead, int64_t iter,
                           sstl::not_null<EntryVec *> outputs, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Cleanup iterations of this frame starting from iteration iter.
        bool CleanupIterations(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);
    };

    // A tagged node: <frame*, iter, node*>.
    struct TaggedNode
    {
        const tf::Node *node = nullptr;
        FrameState *input_frame = nullptr;
        int64_t input_iter = -1;
        bool is_dead = false;
    };

    // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
    // have that many nodes in the ready queue, so we just use a vector and
    // don't free up memory from the queue as we consume nodes.
    class TaggedNodeReadyQueue
    {
    public:
        TaggedNodeReadyQueue()
            : front_index_(0)
        {
        }

        void push_back(TaggedNode node)
        {
            ready_.push_back(node);
        }
        TaggedNode front() const
        {
            DCHECK_LT(front_index_, ready_.size());
            return ready_[front_index_];
        }
        void pop_front()
        {
            DCHECK_LT(front_index_, ready_.size());
            front_index_++;
            if ((front_index_ == ready_.size()) || (front_index_ > 16384)) {
                if (front_index_ == ready_.size()) {
                    ready_.clear();
                } else {
                    // Lots of unused entries at beginning of vector: move everything down
                    // to start of vector.
                    ready_.erase(ready_.begin(), ready_.begin() + front_index_);
                }
                front_index_ = 0;
            }
        }
        bool empty() const
        {
            return ready_.empty();
        }
        const TaggedNode *begin() const
        {
            return ready_.begin() + front_index_;
        }
        const TaggedNode *end() const
        {
            return ready_.end();
        }

    private:
        gtl::InlinedVector<TaggedNode, 16> ready_;
        size_t front_index_;
    };

    struct AsyncState;

    const bool vlog_; // true if VLOG_IS_ON(1). Used to check vlog cheaply.

    std::shared_ptr<IterationContext> ictx_;

    int64_t step_id_;
    // Not owned.
    tf::Rendezvous *rendezvous_;
    tf::SessionState *session_state_;
    tf::TensorStore *tensor_store_;
    // Step-local container.
    tf::ScopedStepContainer *step_container_;
    tf::StepStatsCollector *stats_collector_;
    tf::CallFrameInterface *call_frame_;
    SIExecutor &impl_;
    tf::CancellationManager *cancellation_manager_;
    tf::Executor::Args::Runner runner_;
    bool sync_on_finish_;

    // Owned.

    // step-local container
    tf::checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_;

    // A flag that is set on error after the frame state has been
    // dumped for diagnostic purposes.
    bool dumped_on_error_ = false;

    // The root frame in which the execution of this step is started.
    FrameState *root_frame_;

    // Invoked when the execution finishes.
    tf::Executor::DoneCallback done_cb_;

    std::atomic_int_fast32_t num_outstanding_ops_;

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
    Status PrepareInputs(const NodeItem &item, Entry *first_input, sstl::not_null<TensorValueVec *> inputs,
                         sstl::not_null<DeviceContextVec *> input_device_contexts,
                         sstl::not_null<AllocatorAttributeVec *> input_alloc_attrs, bool *is_input_dead);

    // After item->kernel computation is done, processes its outputs.
    Status ProcessOutputs(const NodeItem &item, tf::OpKernelContext &ctx, sstl::not_null<EntryVec *> outputs);

    // After processing the outputs, propagates the outputs to their dsts.
    // Contents of *outputs are left in an indeterminate state after
    // returning from this method.
    void PropagateOutputs(const TaggedNode &tagged_node, const NodeItem &item,
                          sstl::not_null<EntryVec *> outputs, TaggedNodeSeq *ready);

    // "node" just finishes. Takes ownership of "stats". Returns true if
    // execution has completed.
    bool NodeDone(const Status &s, const tf::Node *node, const TaggedNodeSeq &ready,
                  TaggedNodeReadyQueue *inline_ready);

    // Schedule all the expensive nodes in 'ready', and put all the inexpensive
    // nodes in 'ready' into 'inline_ready'.
    void ScheduleReady(const TaggedNodeSeq &ready, TaggedNodeReadyQueue *inline_ready);

    // Clean up when this executor is done.
    void Finish();

    // For debugging/logging only.
    inline void MaybeMarkCompleted(FrameState *frame, int64_t iter, int id)
    {
        if (vlog_ && VLOG_IS_ON(1)) {
            const auto *item = impl_.gview_.node(id);
            auto l = sstl::with_guard(frame->mu);
            frame->GetIteration(iter)->mark_completed(item->pending_id);
        }
    }

    // A standalone routine for this expression so that we can express
    // that we don't want thread safety analysis on this reference (it's
    // safe to do without the lock because the iterations array never
    // resizes and this particular iteration's array element will not
    // be changed out from under us because the iteration is still alive).
    Entry *GetInputTensors(FrameState *input_frame, int64_t input_iter) const NO_THREAD_SAFETY_ANALYSIS
    {
        return input_frame->GetIteration(input_iter)->input_tensors.data();
    }
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_SIMPLE_ITER_EXECUTORTASK_H
