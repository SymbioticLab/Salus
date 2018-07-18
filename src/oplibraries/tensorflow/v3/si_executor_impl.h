//
// Created by peifeng on 7/18/18.
//

#ifndef SALUS_OPLIB_TENSORFLOW_SIMPLE_ITER_EXECUTOR_IMPL_H
#define SALUS_OPLIB_TENSORFLOW_SIMPLE_ITER_EXECUTOR_IMPL_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "oplibraries/tensorflow/v2/graphview.h"
#include "oplibraries/tensorflow/v3/si_executor.h"

#include <atomic>
#include <memory>

namespace gtl = ::tensorflow::gtl;

namespace salus::oplib::tensorflow {

struct FrameInfo
{
    // The total number of inputs to a frame.
    int input_count = 0;

    // The total number of input tensors of a frame.
    // == sum(nodes[*].num_inputs()) where nodes are the nodes in the frame.
    int total_inputs = 0;

    // Used to determine the next place to allocate space in the
    // pending_counts data structure we'll eventually construct
    tf::PendingCounts::Layout pending_counts_layout;

    // Each frame has its own PendingCounts only for the nodes in the frame.
    std::unique_ptr<tf::PendingCounts> pending_counts = nullptr;

    // The nodes in a frame. Used only for debugging.
    std::vector<const tf::Node *> nodes;
};

struct ControlFlowInfo
{
    gtl::FlatSet<std::string> unique_frame_names;
    std::vector<std::string> frame_names;

    Status Build(const tf::Graph &g);
};

class SIExecutor : public tf::Executor
{
public:
    SIExecutor(SIExecutorParams &&p, std::unique_ptr<const tf::Graph> &&g);

    ~SIExecutor() override;

    Status Initialize();

    void RunAsync(const Args &args, DoneCallback done) override;

private:
    friend class SIExecutorTask;
    friend class SIExecutorState;

    FrameInfo &EnsureFrameInfo(const std::string &fname);
    void InitializePending(const tf::Graph &graph, const ControlFlowInfo &cf_info);

    SIExecutorParams params_;
    std::unique_ptr<const tf::Graph> graph_;
    GraphView gview_;

    // A cached value of params_
    bool device_record_tensor_accesses_ = false;

    // Root nodes (with no in edges) that should form the initial ready queue
    std::vector<const tf::Node *> root_nodes_;

    // Mapping from frame name to static information about the frame.
    // TODO(yuanbyu): We could cache it along with the graph so to avoid
    // the overhead of constructing it for each executor instance.
    gtl::FlatMap<std::string, FrameInfo> frame_info_;

    bool is_main_iter;
    // a combination of graphHandle and partition
    const uint64_t graph_id_;

    static std::atomic_int_fast64_t NextSeq;

    SALUS_DISALLOW_COPY_AND_ASSIGN(SIExecutor);
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_SIMPLE_ITER_EXECUTOR_IMPL_H
