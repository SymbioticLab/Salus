//
// Created by peifeng on 7/19/18.
//

#include "oplibraries/tensorflow/v3/si_executortask.h"

#include "execution/engine/iterationcontext.h"
#include "oplibraries/tensorflow/tfinstance.h"

namespace salus::oplib::tensorflow {

namespace {

// 1-D, 0 element tensor.
const tf::Tensor kEmptyTensor;

bool IsInitializationOp(const tf::Node *node)
{
    return node->op_def().allows_uninitialized_input();
}

void ExecutionEngineRunner(tf::Executor::Args::Closure c)
{
    ExecutionEngine::instance().pool().run(std::move(c));
}

} // namespace

SIExecutorTask::SIExecutorTask(SIExecutor &impl, const tf::Executor::Args &args,
                               tf::Executor::DoneCallback done)
    : m_impl(impl)
    , m_cm(*args.cancellation_manager)
    , m_state(std::make_unique<SIExecutorState>(impl, args, std::move(done)))
{
}

bool SIExecutorTask::prepare()
{
    if (!m_impl.is_main_iter) {
        return true;
    }

    auto &ectx = m_impl.params_.ins;
    return ectx->m_item->beginIteration(ectx->m_ticket, {}, graphId());
}

void SIExecutorTask::runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept
{
    // SIExecutorState will delete itself after called runAsync
    m_state.release()->runAsync(std::move(ictx));
}

SIExecutorState::SIExecutorState(SIExecutor &impl, const tf::Executor::Args &args,
                                 tf::Executor::DoneCallback done)
    : vlog_(VLOG_IS_ON(1))
    , step_id_(args.step_id)
    , rendezvous_(args.rendezvous)
    , session_state_(args.session_state)
    , tensor_store_(args.tensor_store)
    , step_container_(args.step_container)
    , stats_collector_(args.stats_collector)
    , call_frame_(args.call_frame)
    , impl_(impl)
    , cancellation_manager_(args.cancellation_manager)
    , runner_(ExecutionEngineRunner)
    , sync_on_finish_(args.sync_on_finish)
    , done_cb_(std::move(done))
    , num_outstanding_ops_(0)
{
    // We start the entire execution in iteration 0 of the root frame
    // so let us create the root frame and the state for iteration 0.
    // We assume root_frame_->frame_name.empty().
    root_frame_ = new FrameState(impl_, 1);
    root_frame_->frame_id = 0; // must be 0
    root_frame_->InitializeFrameInfo(root_frame_->frame_name);

    // Initialize iteration 0.
    root_frame_->iterations.resize(static_cast<size_t>(root_frame_->max_parallel_iterations));
    root_frame_->iterations[0] =
        std::make_unique<IterationState>(*root_frame_->pending_counts, root_frame_->total_input_tensors);

    outstanding_frames_.insert({root_frame_->frame_name, root_frame_});
}

SIExecutorState::~SIExecutorState()
{
    for (auto name_frame : outstanding_frames_) {
        delete name_frame.second;
    }

    for (auto it : device_context_map_) {
        it->Unref();
    }
}

void SIExecutorState::runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept
{
    VLOG(3) << "ExecutorState::RunAsync for graph@" << as_hex(impl_.graph_)
            << " of graphHandle=" << impl_.params_.graphHandle << " for iter graphId=" << impl_.graph_id_;

    ictx_ = std::move(ictx);
    ictx_->setGraphId(impl_.graph_id_);

    // Ask the device to fill in the device context map.
    auto device = impl_.params_.device;
    auto fill_status = device->FillContextMap(impl_.graph_.get(), &device_context_map_);
    if (!fill_status.ok()) {
        done_cb_(fill_status);
        return;
    }

    // Initialize the ready queue.
    TaggedNodeSeq ready;
    for (const auto *n : impl_.root_nodes_) {
        DCHECK(n->in_edges().empty());
        ready.push_back(TaggedNode{n, root_frame_, 0, false});
    }
    if (ready.empty()) {
        done_cb_(Status::OK());
    } else {
        num_outstanding_ops_ = ready.size();
        root_frame_->iterations[0]->outstanding_ops = static_cast<int>(ready.size());
        // Schedule to run all the ready ops in thread pool.
        ScheduleReady(ready, nullptr);
    }
}

// State kept alive for executing an asynchronous node in another
// thread.  NOTE: We need to make a copy of p.input,
// p.input_device_contexts, and p.input_alloc_attrs for asynchronous
// kernels because OpKernelContext methods like input_type(i) needs
// the param points to valid input type vector. It's not an issue for
// sync kernels because these vectors are kept on the stack.
struct SIExecutorState::AsyncState
{
    AsyncState(const tf::OpKernelContext::Params &p, const TaggedNode &_tagged_node, const NodeItem *_item,
               Entry *_first_input)
        : saved_inputs(*p.inputs)
        , saved_input_device_contexts(*p.input_device_contexts)
        , saved_input_alloc_attrs(*p.input_alloc_attrs)
        , params(p)
        , tagged_node(_tagged_node)
        , item(_item)
        , first_input(_first_input)
        ,
        // ParamsButClearingEigenGPUDevice does equivalent of
        //   params.eigen_gpu_device = nullptr;
        ctx(ParamsButClearingEigenGPUDevice(&params), item->num_outputs)
    {
        params.inputs = &saved_inputs;
        params.input_device_contexts = &saved_input_device_contexts;
        params.input_alloc_attrs = &saved_input_alloc_attrs;
    }

    TensorValueVec saved_inputs;
    DeviceContextVec saved_input_device_contexts;
    AllocatorAttributeVec saved_input_alloc_attrs;
    tf::OpKernelContext::Params params;
    TaggedNode tagged_node;
    const NodeItem *item;
    Entry *first_input;
    tf::OpKernelContext ctx;

private:
    tf::OpKernelContext::Params *ParamsButClearingEigenGPUDevice(tf::OpKernelContext::Params *p)
    {
        // Ensure OpKernelContext constructor will make a new eigen GPU device if
        // necessary.
        p->eigen_gpu_device = nullptr; // Force allocation
        return p;
    }
};

void SIExecutorState::Process(SIExecutorState::TaggedNode tagged_node)
{
    const auto &gview = impl_.gview_;
    TaggedNodeSeq ready;
    TaggedNodeReadyQueue inline_ready;

    // Parameters passed to OpKernel::Compute.
    TensorValueVec inputs;
    DeviceContextVec input_device_contexts;
    AllocatorAttributeVec input_alloc_attrs;

    tf::OpKernelContext::Params params;
    params.step_id = step_id_;
    auto device = impl_.params_.device;
    params.device = device;
    params.record_tensor_accesses = impl_.device_record_tensor_accesses_;
    params.rendezvous = rendezvous_;
    params.session_state = session_state_;
    params.tensor_store = tensor_store_;
    params.cancellation_manager = cancellation_manager_;
    params.call_frame = call_frame_;
    params.function_library = impl_.params_.function_library;
    params.resource_manager = device->resource_manager();
    params.step_container = step_container_;
    params.slice_reader_cache = &slice_reader_cache_;
    params.inputs = &inputs;
    params.input_device_contexts = &input_device_contexts;
    params.input_alloc_attrs = &input_alloc_attrs;
    params.runner = &runner_;
    params.stats_collector = stats_collector_;

    Status s;
    EntryVec outputs;
    bool completed = false;
    inline_ready.push_back(tagged_node);
    while (!inline_ready.empty()) {
        tagged_node = inline_ready.front();
        inline_ready.pop_front();
        const auto *node = tagged_node.node;
        auto input_frame = tagged_node.input_frame;
        const auto input_iter = tagged_node.input_iter;
        const int id = node->id();
        const auto &item = *gview.node(id);

        // TODO(misard) Replace with a finer-grain enabling flag once we
        // add better optional debugging support.
        if (vlog_ && VLOG_IS_ON(1)) {
            auto l = sstl::with_guard(input_frame->mu);
            input_frame->GetIteration(input_iter)->mark_started(item.pending_id);
        }

        // Set the device_context for this node id, if it exists.
        if (static_cast<size_t>(id) < device_context_map_.size()) {
            params.op_device_context = device_context_map_[id];
        }

        params.track_allocations = false;

        if (vlog_) {
            VLOG(1) << "Process node: " << id << " step " << params.step_id << " " << SummarizeNode(*node)
                    << " is dead: " << tagged_node.is_dead;
        }

        auto input_tensors = GetInputTensors(input_frame, input_iter);
        auto first_input = input_tensors + item.input_start;
        outputs.clear();

        tf::TensorReferenceVector accessed_tensors;
        tf::DeviceContext *device_context = nullptr;
        // Only execute this node if it is not dead or it is a send/recv
        // transfer node. For transfer nodes, we need to propagate the "dead"
        // bit even when the node is dead.
        bool launched_asynchronously = false;
        if (tagged_node.is_dead && !IsTransferNode(node)) {
            outputs.resize(static_cast<size_t>(item.num_outputs));
        } else {
            // Prepares inputs.
            bool is_input_dead = false;
            s = PrepareInputs(item, first_input, &inputs, &input_device_contexts, &input_alloc_attrs,
                              &is_input_dead);
            if (!s.ok()) {
                // Clear inputs.
                int num_inputs = item.num_inputs;
                for (int i = 0; i < num_inputs; ++i) {
                    first_input[i].ClearVal();
                }
                MaybeMarkCompleted(input_frame, input_iter, id);
                // Continue to process the nodes in 'inline_ready'.
                completed = NodeDone(s, item.node, ready, &inline_ready);
                continue;
            }

            // Set up compute params.
            auto op_kernel = item.kernel.get();
            params.op_kernel = op_kernel;
            params.frame_iter = tf::FrameAndIter(input_frame->frame_id, input_iter);
            params.is_input_dead = is_input_dead;
            params.output_attr_array = item.output_attrs();

            if (item.kernel_is_async) {
                // Asynchronous computes.
                auto async = item.kernel->AsAsync();
                DCHECK(async != nullptr);
                launched_asynchronously = true;
                auto state = new AsyncState(params, tagged_node, &item, first_input);

                auto done = [this, state]() {
                    auto device = impl_.params_.device;
                    auto first_input = state->first_input; // Shorthand

                    EntryVec outputs;
                    auto s = ProcessOutputs(*state->item, state->ctx, &outputs);
                    if (vlog_) {
                        VLOG(2) << "Async kernel done: " << state->item->node->id() << " step " << step_id_
                                << " " << SummarizeNode(*state->item->node)
                                << " is dead: " << state->tagged_node.is_dead;
                    }
                    // Clears inputs.
                    const int num_inputs = state->item->num_inputs;
                    for (int i = 0; i < num_inputs; ++i) {
                        first_input[i].ClearVal();
                    }
                    auto input_frame = state->tagged_node.input_frame;
                    const auto input_iter = state->tagged_node.input_iter;
                    const int id = state->tagged_node.node->id();
                    MaybeMarkCompleted(input_frame, input_iter, id);
                    TaggedNodeSeq ready;
                    if (s.ok()) {
                        PropagateOutputs(state->tagged_node, *state->item, &outputs, &ready);
                    }
                    outputs.clear();
                    if (s.ok() && impl_.device_record_tensor_accesses_) {
                        // Get the list of all tensors accessed during the execution
                        tf::TensorReferenceVector accessed;
                        state->ctx.retrieve_accessed_tensors(&accessed);
                        // callee takes ownership of the vector
                        device->ConsumeListOfAccessedTensors(state->ctx.op_device_context(), accessed);
                    }
                    const bool completed = NodeDone(s, state->item->node, ready, nullptr);
                    delete state;
                    if (completed) {
                        Finish();
                    }
                };
                device->ComputeAsync(async, &state->ctx, done);
            } else {
                // Synchronous computes.
                tf::OpKernelContext ctx(&params, item.num_outputs);
                CHECK_NOTNULL(op_kernel);
                device->Compute(op_kernel, &ctx);
                s = ProcessOutputs(item, ctx, &outputs);
                if (s.ok() && impl_.device_record_tensor_accesses_) {
                    // Get the list of all tensors accessed during the execution
                    ctx.retrieve_accessed_tensors(&accessed_tensors);
                    device_context = ctx.op_device_context();
                }
            }
        }

        if (!launched_asynchronously) {
            if (vlog_) {
                VLOG(2) << "Synchronous kernel done: " << id << " step " << params.step_id << " "
                        << SummarizeNode(*node) << " is dead: " << tagged_node.is_dead;
            }

            // Clears inputs.
            const int num_inputs = item.num_inputs;
            for (int i = 0; i < num_inputs; ++i) {
                first_input[i].ClearVal();
            }
            MaybeMarkCompleted(input_frame, input_iter, id);
            // Propagates outputs.
            if (s.ok()) {
                PropagateOutputs(tagged_node, item, &outputs, &ready);
            }
            outputs.clear();
            if (!accessed_tensors.empty()) {
                // device_context is set above in synchronous computes
                device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
            }
            // Postprocess.
            completed = NodeDone(s, item.node, ready, &inline_ready);
        }
    } // while !inline_ready.empty()

    // This thread of computation is done if completed = true.
    if (completed) {
        Finish();
    }
}

Status SIExecutorState::PrepareInputs(const NodeItem &item, Entry *first_input,
                                      sstl::not_null<TensorValueVec *> inputs,
                                      sstl::not_null<DeviceContextVec *> input_device_contexts,
                                      sstl::not_null<AllocatorAttributeVec *> input_alloc_attrs,
                                      bool *is_input_dead)
{
    auto node = item.node;

    sstl::static_for(std::tie(inputs, input_device_contexts, input_alloc_attrs), [&](auto, auto parr) {
        parr->clear();
        parr->resize(static_cast<size_t>(item.num_inputs));
    });

    *is_input_dead = false;

    bool is_merge = item.is_merge;
    for (int i = 0; i < item.num_inputs; ++i) {
        const bool expect_ref = IsRefType(item.input_type(i));
        auto &entry = first_input[i];

        (*input_device_contexts)[i] = entry.device_context;
        (*input_alloc_attrs)[i] = entry.alloc_attr;

        // i-th input.
        auto &inp = (*inputs)[i];

        // Only merge and transfer nodes can have no-value inputs.
        if (!entry.has_value) {
            if (!is_merge) {
                DCHECK(IsTransferNode(node)) << node->name() << " - input " << i;
                DCHECK(!entry.val_field_is_set) << node->name() << " - input " << i;
                entry.SetVal(kEmptyTensor);
                inp.tensor = entry.val.get();
                *is_input_dead = true;
            }
            continue;
        }
        if (entry.ref == nullptr) {
            if (expect_ref) {
                return AttachDef(tf::errors::InvalidArgument(i, "-th input expects a ref type"),
                                 item.kernel->def());
            }
        } else {
            {
                Entry::MaybeLock ml(&entry);
                if (!entry.ref->IsInitialized() && !IsInitializationOp(item.node)) {
                    return AttachDef(tf::errors::FailedPrecondition("Attempting to use uninitialized value ",
                                                                    item.kernel->requested_input(i)),
                                     item.kernel->def());
                }
            }
            if (expect_ref) {
                inp.mutex_if_ref = entry.ref_mu;
            } else {
                // Automatically deref the tensor ref when the op expects a
                // tensor but is given a ref to a tensor. Need to deref it
                // under the mutex.
                entry.Dereference();
            }
        }
        inp.tensor = entry.RefOrVal();
    }
    return Status::OK();
}

Status SIExecutorState::ProcessOutputs(const NodeItem &item, tf::OpKernelContext &ctx,
                                       sstl::not_null<EntryVec *> outputs)
{
    auto node = item.node;
    DCHECK(outputs->empty());
    outputs->resize(static_cast<size_t>(item.num_outputs));

    auto s = ctx.status();
    if (!s.ok()) {
        s = AttachDef(s, node->def());
        return s;
    }

    // Get the device_context for this node id, if it exists.
    auto device_context = ctx.op_device_context();

    // Experimental: debugger (tfdb) access to intermediate node completion.
    if (item.num_outputs == 0 && impl_.params_.node_outputs_cb != nullptr) {
        // If the node has no output, invoke the callback with output slot set to
        // -1, signifying that this is a no-output node.
        s.Update(impl_.params_.node_outputs_cb(item.node->name(), -1, nullptr, false, &ctx));
    }

    for (int i = 0; i < item.num_outputs; ++i) {
        auto val = ctx.release_output(i);
        if (*ctx.is_output_dead() || !val.tensor) {
            // Unless it's a Switch or a Recv, the node must produce a
            // tensor value at i-th output.
            if (!IsSwitch(node) && !IsRecv(node)) {
                s.Update(
                    tf::errors::Internal("Missing ", i, "-th output from ", SummarizeNodeDef(node->def())));
            }
        } else {
            auto &out = (*outputs)[i];

            // Set the device of the output entry.
            out.device_context = device_context;

            // Set the allocator attributes of the output entry.
            out.alloc_attr = ctx.output_alloc_attr(i);

            // Sanity check of output tensor types.
            tf::DataType dtype;
            if (val.is_ref()) {
                tf::mutex_lock ml(*val.mutex_if_ref);
                dtype = MakeRefType(val->dtype());
            } else {
                dtype = val->dtype();
            }
            if (dtype == item.output_type(i)) {
                if (val.is_ref()) {
                    out.has_value = true;
                    out.ref = val.tensor;
                    out.ref_mu = val.mutex_if_ref;

                    // Experimental: debugger (tfdb) access to intermediate node
                    // outputs.
                    if (impl_.params_.node_outputs_cb != nullptr) {
                        s.Update(impl_.params_.node_outputs_cb(item.node->name(), i, out.ref, true, &ctx));
                    }
                } else {
                    // NOTE that std::move is used here, so val.tensor goes to
                    // uninitialized state (val.tensor->IsInitialized return false).
                    DCHECK(!out.val_field_is_set);
                    out.has_value = true;
                    out.val_field_is_set = true;
                    out.val.Init(std::move(*val.tensor));

                    // Experimental: debugger access to intermediate node outputs.
                    if (impl_.params_.node_outputs_cb != nullptr) {
                        s.Update(
                            impl_.params_.node_outputs_cb(item.node->name(), i, out.val.get(), false, &ctx));
                    }
                }
            } else {
                s.Update(tf::errors::Internal("Output ", i, " of type ", DataTypeString(dtype),
                                              " does not match declared output type ",
                                              DataTypeString(item.output_type(i)), " for node ",
                                              SummarizeNode(*node)));
            }
        }

        if (!val.is_ref()) {
            // If OpKernelContext returns outputs via pass-by-value, we
            // don't need this trouble.
            delete val.tensor;
        }
    }
    return s;
}

void SIExecutorState::PropagateOutputs(const SIExecutorState::TaggedNode &tagged_node, const NodeItem &item,
                                       sstl::not_null<EntryVec *> outputs, TaggedNodeSeq *ready)
{
    const auto node = tagged_node.node;
    auto input_frame = tagged_node.input_frame;
    const auto input_iter = tagged_node.input_iter;
    const bool is_dead = tagged_node.is_dead;

    // Propagates outputs along out edges, and puts newly ready nodes
    // into the ready queue.
    ready->clear();
    bool is_frame_done;
    auto output_frame = input_frame;
    auto output_iter = input_iter;

    if (!item.is_enter_exit_or_next_iter) {
        // Fast path for nodes types that don't need special handling
        DCHECK_EQ(input_frame, output_frame);
        // Normal path for most nodes
        auto l = sstl::with_guard(input_frame->mu);
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
        is_frame_done = input_frame->DecrementOutstandingOpsLocked(impl_.gview_, input_iter, ready);
    } else if (item.is_enter) {
        bool is_constant;
        auto s = GetNodeAttr(node->def(), "is_constant", &is_constant);
        DCHECK(s.ok()) << s;
        FindOrCreateChildFrame(input_frame, input_iter, node, &output_frame);
        output_iter = 0;
        {
            auto l = sstl::with_guard(output_frame->mu);
            if (is_constant) {
                // Propagate to all active iterations if this is a loop invariant.
                output_frame->AddLoopInv(item, (*outputs)[0], ready);
            } else {
                output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
            }
            output_frame->num_pending_inputs--;
        }
        is_frame_done = input_frame->DecrementOutstandingOps(impl_.gview_, input_iter, ready);
    } else if (item.is_exit) {
        if (is_dead) {
            auto l = sstl::with_guard(input_frame->mu);
            // Stop and remember this node if it is a dead exit.
            if (input_iter == input_frame->iteration_count) {
                input_frame->dead_exits.push_back(node);
            }
            is_frame_done = input_frame->DecrementOutstandingOpsLocked(impl_.gview_, input_iter, ready);
        } else {
            output_frame = input_frame->parent_frame;
            output_iter = input_frame->parent_iter;
            {
                auto l = sstl::with_guard(output_frame->mu);
                output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
            }
            is_frame_done = input_frame->DecrementOutstandingOps(impl_.gview_, input_iter, ready);
        }
    } else {
        DCHECK(IsNextIteration(node));
        auto l = sstl::with_guard(input_frame->mu);
        if (is_dead) {
            // Stop the deadness propagation.
            output_frame = nullptr;
        } else {
            if (input_iter == input_frame->iteration_count
                && input_frame->num_outstanding_iterations == input_frame->max_parallel_iterations) {
                // Reached the maximum for parallel iterations.
                input_frame->next_iter_roots.emplace_back(node, (*outputs)[0]);
                output_frame = nullptr;
            } else {
                // If this is a new iteration, start it.
                if (input_iter == input_frame->iteration_count) {
                    input_frame->IncrementIteration(impl_.gview_, ready);
                }
                output_iter = input_iter + 1;
            }
        }
        if (output_frame != nullptr) {
            // This is the case when node is not Enter, Exit, or NextIteration.
            DCHECK(input_frame == output_frame);
            output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
        }
        is_frame_done = input_frame->DecrementOutstandingOpsLocked(impl_.gview_, input_iter, ready);
    }

    // At this point, this node is completely done. We also know if the
    // completion of this node makes its frame completed.
    if (is_frame_done) {
        auto parent_frame = input_frame->parent_frame;
        const auto parent_iter = input_frame->parent_iter;
        DeleteFrame(input_frame, ready);
        if (parent_frame != nullptr) {
            // The completion of frame may cause completions in its parent frame.
            // So clean things up recursively.
            CleanupFramesIterations(parent_frame, parent_iter, ready);
        }
    }
}

bool SIExecutorState::NodeDone(const Status &s, const tf::Node *node,
                               const SIExecutorState::TaggedNodeSeq &ready,
                               SIExecutorState::TaggedNodeReadyQueue *inline_ready)
{
    VLOG(1) << "NodeDone: " << node->id() << " step " << step_id_ << " " << SummarizeNode(*node);
    if (VLOG_IS_ON(1)) {
        for (auto &tn : ready) {
            VLOG(1) << "NodeReady: " << tn.node->id() << " step " << step_id_ << " "
                    << SummarizeNode(*tn.node);
        }
    }

    bool abort_run = false;
    if (!s.ok()) {
        // Some error happened. This thread of computation is done.
        auto l = sstl::with_guard(mu_);
        if (status_.ok()) {
            abort_run = true;
            status_ = s;
        }
    }
    if (abort_run) {
        if (rendezvous_) {
            rendezvous_->StartAbort(s);
        }
        if (cancellation_manager_) {
            cancellation_manager_->StartCancel();
        }
    }

    bool completed = false;
    const auto ready_size = ready.size();
    if (ready_size == 0 || !s.ok()) {
        completed = (num_outstanding_ops_.fetch_sub(1) == 1);
    } else if (ready_size > 1) {
        num_outstanding_ops_.fetch_add(ready_size - 1, std::memory_order_relaxed);
    }

    // Schedule the ready nodes in 'ready'.
    if (s.ok()) {
        ScheduleReady(ready, inline_ready);
    }
    return completed;
}

void SIExecutorState::ScheduleReady(const TaggedNodeSeq &ready, TaggedNodeReadyQueue *inline_ready)
{
    if (ready.empty()) {
        return;
    }

    if (inline_ready == nullptr) {
        // Schedule to run all the ready ops in thread pool.
        for (auto &tagged_node : ready) {
            runner_([=]() { Process(tagged_node); });
        }
        return;
    }
    const auto &gview = impl_.gview_;
    const TaggedNode *curr_expensive_node = nullptr;
    for (auto &tagged_node : ready) {
        const auto &item = *gview.node(tagged_node.node->id());
        if (tagged_node.is_dead || !item.kernel_is_expensive) {
            // Inline this inexpensive node.
            inline_ready->push_back(tagged_node);
        } else {
            if (curr_expensive_node) {
                // Dispatch to another thread since there is plenty of work to
                // do for this thread.
                runner_([this, curr_expensive_node]() { Process(*curr_expensive_node); });
            }
            curr_expensive_node = &tagged_node;
        }
    }

    if (curr_expensive_node) {
        if (inline_ready->empty()) {
            // Tail recursion optimization
            inline_ready->push_back(*curr_expensive_node);
        } else {
            // There are inline nodes to run already. We dispatch this expensive
            // node to other thread.
            runner_([this, curr_expensive_node]() { Process(*curr_expensive_node); });
        }
    }
}

void SIExecutorState::Finish()
{
    auto l = sstl::with_uguard(mu_);
    auto status = status_;
    auto done_cb = std::move(done_cb_);
    auto runer = std::move(runner_);
    auto ictx = std::move(ictx_);
    l.unlock();

    if (sync_on_finish_ && status.ok()) {
        status = impl_.params_.device->Sync();
    }

    if (impl_.is_main_iter) {
        impl_.params_.ins->dropExlusiveMode();
        LogOpTracing() << "event: end_iter "
                       << nlohmann::json({{"sess", impl_.params_.session},
                                          {"graphId", impl_.graph_id_},
                                          {"memMap", TFInstance::instance().dumpGPUMemoryMap()}});
        ictx->finish();
    }

    delete this;
    CHECK(done_cb != nullptr);
    runner_([done = std::move(done_cb), status]() { done(status); });
}

void SIExecutorState::FindOrCreateChildFrame(FrameState *frame, int64_t iter, const tf::Node *node,
                                             FrameState **child)
{
    // Get the child frame name.
    std::string enter_name;
    auto s = GetNodeAttr(node->attrs(), "frame_name", &enter_name);
    DCHECK(s.ok()) << s;

    const auto child_name = MakeFrameName(frame, iter, enter_name);

    {
        auto executor_lock = sstl::with_guard(mu_);
        *child = sstl::getOrDefault(outstanding_frames_, child_name, nullptr);
        if (*child) {
            return;
        }
    }

    // Need to create a new frame instance.
    // Note that this new frame instance is created without any locks.
    if (vlog_) {
        VLOG(2) << "Create frame: " << child_name;
    }

    int parallel_iters;
    s = GetNodeAttr(node->attrs(), "parallel_iterations", &parallel_iters);
    DCHECK(s.ok()) << s;
    auto temp = new FrameState(impl_, parallel_iters);
    temp->frame_name = child_name;
    temp->frame_id = tf::Hash64(child_name);
    temp->parent_frame = frame;
    temp->parent_iter = iter;
    temp->InitializeFrameInfo(enter_name);

    // 'iterations' is a fixed-length circular buffer.
    temp->iterations.resize(temp->max_parallel_iterations + 1_sz);
    // Initialize iteration 0.
    temp->iterations[0] = std::make_unique<IterationState>(*temp->pending_counts, temp->total_input_tensors);

    {
        auto l = sstl::with_guard(mu_);
        auto it = outstanding_frames_.find(child_name);
        if (it != outstanding_frames_.end()) {
            *child = it->second;
        } else {
            auto frame_lock = sstl::with_guard(frame->mu);
            frame->GetIteration(iter)->outstanding_frame_count++;
            outstanding_frames_[child_name] = temp;
            *child = temp;
            temp = nullptr;
        }
    }
    delete temp; // Not used so delete it.
}

void SIExecutorState::DeleteFrame(FrameState *frame, TaggedNodeSeq *ready)
{
    // First, propagate dead_exits (if any) to the parent frame.
    auto parent_frame = frame->parent_frame;
    const auto parent_iter = frame->parent_iter;
    if (parent_frame) {
        auto parent_frame_lock = sstl::with_guard(parent_frame->mu);
        // Propagate all the dead exits to the parent frame.
        for (auto *node : frame->dead_exits) {
            auto parent_iter_state = parent_frame->GetIteration(parent_iter);
            for (auto *e : node->out_edges()) {
                auto *dst_node = e->dst();

                auto dst_pending_id = impl_.gview_.node(dst_node->id())->pending_id;

                // TODO(yuanbyu): We don't need this if we require the subgraph
                // given to an executor not to contain a sink node.
                if (dst_node->IsSink()) {
                    continue;
                }

                bool dst_dead = true;
                bool dst_ready = false;
                // We know this is a dead input to dst.
                if (tf::IsMerge(dst_node)) {
                    if (e->IsControlEdge()) {
                        parent_iter_state->decrement_pending(dst_pending_id, 2);
                        int count = parent_iter_state->pending(dst_pending_id);
                        int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
                        dst_dead = (dead_cnt == dst_node->num_inputs());
                        dst_ready = (count == 0) || ((count == 1) && dst_dead);
                    } else {
                        parent_iter_state->increment_dead_count(dst_pending_id);
                        const int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
                        dst_dead = (dead_cnt == dst_node->num_inputs());
                        dst_ready = (parent_iter_state->pending(dst_pending_id) == 1) && dst_dead;
                    }
                } else {
                    parent_iter_state->increment_dead_count(dst_pending_id);
                    dst_ready = (parent_iter_state->decrement_pending(dst_pending_id, 1) == 0);
                }
                if (dst_ready) {
                    if (tf::IsControlTrigger(dst_node)) {
                        dst_dead = false;
                    }
                    ready->push_back(TaggedNode{dst_node, parent_frame, parent_iter, dst_dead});
                    parent_iter_state->outstanding_ops++;
                }
            }
        }
    }

    // Delete the frame.
    const auto &frame_name = frame->frame_name;
    if (vlog_) {
        VLOG(2) << "Delete frame " << frame_name;
    }
    {
        auto l = sstl::with_guard(mu_);
        outstanding_frames_.erase(frame_name);
    }
    delete frame;
}

void SIExecutorState::CleanupFramesIterations(FrameState *frame, int64_t iter, TaggedNodeSeq *ready)
{
    bool is_frame_done = false;
    {
        auto l = sstl::with_guard(frame->mu);
        frame->GetIteration(iter)->outstanding_frame_count--;
        is_frame_done = frame->CleanupIterations(impl_.gview_, iter, ready);
    }
    if (is_frame_done) {
        auto parent_frame = frame->parent_frame;
        auto parent_iter = frame->parent_iter;
        DeleteFrame(frame, ready);
        if (parent_frame) {
            // The completion of frame may cause completions in its parent frame
            // So clean things up recursively.
            CleanupFramesIterations(parent_frame, parent_iter, ready);
        }
    }
}

void SIExecutorState::FrameState::ActivateNodes(const NodeItem &item, bool is_dead, int64_t iter,
                                                sstl::not_null<EntryVec *> poutputs, TaggedNodeSeq *ready)
{
    auto &gview = executor.gview_;
    auto &outputs = *poutputs;
    auto iter_state = GetIteration(iter);
    auto num_output_edges = item.num_output_edges;
    auto edges = item.output_edge_list();
    auto input_tensors = iter_state->input_tensors;
    for (int out_index = 0; out_index < num_output_edges; ++out_index) {
        const auto &e = edges[out_index];
        const auto dst_id = e.dst_id;
        const auto &dst_item = *gview.node(dst_id);
        const auto dst_pending_id = dst_item.pending_id;
        const auto src_slot = e.output_slot;

        // TODO(yuanbyu): We don't need this if we require the subgraph
        // given to an executor not to contain a sink node.
        if (dst_item.is_sink) {
            continue;
        }

        bool dst_dead = false;
        bool dst_ready = false;
        // True iff this input for dst is needed. We only set this input for
        // dst if this flag is true. This is needed to make the thread safety
        // analysis happy.
        const bool is_control_edge = (src_slot == tf::Graph::kControlSlot);
        bool dst_need_input = !is_control_edge;

        if (dst_item.is_merge) {
            // A merge node is ready if all control inputs have arrived and either
            // a) a live data input becomes available or b) all data inputs are
            // dead. For Merge, pending's LSB is set iff a live data input has
            // arrived.
            if (is_control_edge) {
                iter_state->decrement_pending(dst_pending_id, 2);
                int count = iter_state->pending(dst_pending_id);
                int dead_cnt = iter_state->dead_count(dst_pending_id);
                dst_dead = (dead_cnt == dst_item.num_inputs);
                dst_ready = (count == 0) || ((count == 1) && dst_dead);
            } else {
                if (outputs[src_slot].has_value) {
                    // This is a live data input.
                    int count = iter_state->pending(dst_pending_id);
                    iter_state->mark_live(dst_pending_id);
                    // Only the first live edge sets the input and (potentially)
                    // triggers execution. The low bit of count is set if and
                    // only if no live input has been used yet (mark_live clears
                    // it). The node should be started if and only if this is
                    // the first live input and there are no pending control
                    // edges, i.e. count == 1.
                    dst_ready = (count == 1);
                    dst_need_input = ((count & 0x1) == 1);
                } else {
                    // This is a dead data input. Note that dst_node is dead if node is
                    // a dead enter. We need this to handle properly a while loop on
                    // the untaken branch of a conditional.
                    // TODO(yuanbyu): This is a bit hacky, but a good solution for
                    // now.
                    iter_state->increment_dead_count(dst_pending_id);
                    const int dead_cnt = iter_state->dead_count(dst_pending_id);
                    dst_dead = (dead_cnt == dst_item.num_inputs) || item.is_enter;
                    dst_ready = (iter_state->pending(dst_pending_id) == 1) && dst_dead;
                    dst_need_input = false;
                }
            }
        } else {
            const bool increment_dead = (is_dead || (!is_control_edge && !(outputs)[src_slot].has_value));
            int pending, dead;
            iter_state->adjust_for_activation(dst_pending_id, increment_dead, &pending, &dead);
            dst_dead = (dead > 0);
            dst_ready = (pending == 0);
        }

        if (dst_need_input) {
            const int dst_slot = e.input_slot;
            const int dst_loc = dst_item.input_start + dst_slot;
            if (e.is_last) {
                input_tensors[dst_loc] = std::move(outputs[src_slot]);
            } else {
                input_tensors[dst_loc] = outputs[src_slot];
            }
        }

        if (dst_ready) {
            if (dst_item.is_control_trigger) {
                dst_dead = false;
            }
            ready->push_back(TaggedNode{dst_item.node, this, iter, dst_dead});
            iter_state->outstanding_ops++;
        }
    }
}

void SIExecutorState::FrameState::ActivateNexts(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
{
    // Propagate the deferred NextIteration nodes to the new iteration.
    for (auto &[node, entry] : next_iter_roots) {
        auto is_dead = !entry.has_value;
        auto &item = *gview.node(node->id());
        EntryVec outputs{entry};
        ActivateNodes(item, is_dead, iter, &outputs, ready);
    }
    next_iter_roots.clear();
}

void SIExecutorState::FrameState::ActivateLoopInvs(const GraphView &gview, int64_t iter, TaggedNodeSeq *ready)
{
    // Propagate loop invariants to the new iteration.
    for (auto &[node, entry] : inv_values) {
        auto is_dead = !entry.has_value;
        auto &item = *gview.node(node->id());
        EntryVec outputs{entry};
        ActivateNodes(item, is_dead, iter, &outputs, ready);
    }
}

void SIExecutorState::FrameState::AddLoopInv(const NodeItem &item, const Entry &entry, TaggedNodeSeq *ready)
{
    // Store this value.
    inv_values.emplace_back(item.node, entry);

    // Make this value available to all iterations.
    const bool is_dead = !entry.has_value;
    EntryVec outputs{entry};
    for (int i = 0; i <= iteration_count; ++i) {
        ActivateNodes(item, is_dead, i, &outputs, ready);
    }
}

bool SIExecutorState::FrameState::IsIterationDone(int64_t iter)
{
    auto iter_state = GetIteration(iter);
    if (iter_state->outstanding_ops == 0 && iter_state->outstanding_frame_count == 0) {
        if (iter == 0) {
            // The enclosing frame has no pending input.
            return num_pending_inputs == 0;
        } else {
            // The preceding iteration is deleted (and therefore done).
            return (GetIteration(iter - 1) == nullptr);
        }
    }
    return false;
}

void SIExecutorState::FrameState::IncrementIteration(const GraphView &gview, TaggedNodeSeq *ready)
{
    iteration_count++;
    const auto next_iter = iteration_count;

    // Initialize the next iteration.
    auto iter_state = std::make_unique<IterationState>(*pending_counts, total_input_tensors);
    SetIteration(next_iter, std::move(iter_state));
    num_outstanding_iterations++;
    dead_exits.clear();

    // Activate the successors of the deferred roots in the new iteration.
    ActivateNexts(gview, next_iter, ready);

    // Activate the loop invariants in the new iteration.
    ActivateLoopInvs(gview, next_iter, ready);
}

bool SIExecutorState::FrameState::CleanupIterations(const GraphView &gview, int64_t iter,
                                                    SIExecutorState::TaggedNodeSeq *ready)
{
    auto curr_iter = iter;
    while (curr_iter <= iteration_count && IsIterationDone(curr_iter)) {
        // Delete the iteration curr_iter.
        SetIteration(curr_iter, nullptr);
        --num_outstanding_iterations;
        ++curr_iter;

        // When one iteration is completed, we check for deferred iteration,
        // and start it if there is one.
        if (!next_iter_roots.empty()) {
            IncrementIteration(gview, ready);
        }
    }
    return IsFrameDone();
}
} // namespace salus::oplib::tensorflow
