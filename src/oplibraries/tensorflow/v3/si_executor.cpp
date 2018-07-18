//
// Created by peifeng on 7/18/18.
//

#include "oplibraries/tensorflow/v3/si_executor.h"

#include "oplibraries/tensorflow/v3/si_executor_impl.h"
#include "oplibraries/tensorflow/v3/si_executortask.h"

namespace salus::oplib::tensorflow {

namespace {

/**
 * @brief
 * @param n
 * @return pair of max_pending and max_dead_count
 */
std::pair<size_t, size_t> GetMaxPendingCounts(const tf::Node &n)
{
    const auto num_in_edges = n.in_edges().size();
    size_t initial_count;
    if (IsMerge(&n)) {
        // merge waits all control inputs so we initialize the pending
        // count to be the number of control edges.
        int32_t num_control_edges = 0;
        for (const auto *edge : n.in_edges()) {
            if (edge->IsControlEdge()) {
                num_control_edges++;
            }
        }
        // Use bit 0 to indicate if we are waiting for a ready live data input.
        initial_count = 1_sz + (num_control_edges << 1);
    } else {
        initial_count = num_in_edges;
    }

    return {initial_count, num_in_edges};
}

} // namespace

Status NewSIExecutor(SIExecutorParams params, std::unique_ptr<const tf::Graph> &&graph,
                     tf::Executor **executor)
{
    auto impl = std::make_unique<SIExecutor>(std::move(params), std::move(graph));
    auto s = impl->Initialize();
    if (s.ok()) {
        *executor = impl.release();
    }
    return s;
}

std::atomic_int_fast64_t SIExecutor::NextSeq{0};

SIExecutor::SIExecutor(SIExecutorParams &&p, std::unique_ptr<const tf::Graph> &&g)
    : params_(std::move(p))
    , graph_(std::move(g))
    , is_main_iter(false)
    , graph_id_(static_cast<uint64_t>(++NextSeq))
{
    DCHECK(params_.create_kernel != nullptr);
}

SIExecutor::~SIExecutor() = default;

Status SIExecutor::Initialize()
{
    gview_.Initialize(graph_.get());

    // Build the information about frames in this subgraph.
    ControlFlowInfo cf_info;
    TF_RETURN_IF_ERROR(cf_info.Build(*graph_));

    // Cache this value so we make this virtual function call once, rather
    // that O(# steps * # nodes per step) times.
    device_record_tensor_accesses_ = params_.device->RequiresRecordingAccessedTensors();

    frame_info_.reserve(cf_info.unique_frame_names.size());

    VLOG(2) << "Created graph@" << as_hex(graph_) << " of graphHandle=" << params_.graphHandle
            << ", graphId=" << graph_id_ << " for session " << params_.session;

    // Preprocess every node in the graph to create an instance of op
    // kernel for each node.
    for (const auto *n : graph_->nodes()) {
        const int id = n->id();
        const auto &fname = cf_info.frame_names[id];
        auto &frame_info = EnsureFrameInfo(fname);

        VLOG(3) << "Node " << id << " in graphHandle=" << params_.graphHandle << ", graphId=" << graph_id_
                << ": " << n->def();

        // Check if this is main iteration
        if (n->name() == "salus_main_iter") {
            VLOG(2) << params_.session << ":" << params_.graphHandle << " is main iteration";
            is_main_iter = true;
        }

        // See if this node is a root node, and if so, add to root_nodes_.
        if (n->in_edges().empty()) {
            root_nodes_.push_back(n);
        }

        auto item = gview_.node(id);
        item->node = n;

        item->input_start = frame_info.total_inputs;
        frame_info.total_inputs += n->num_inputs();

        auto s = params_.create_kernel(n->def(), &item->kernel);
        if (!s.ok()) {
            item->kernel = nullptr;
            s = AttachDef(s, *n);
            LOG(ERROR) << "Executor failed to create kernel. " << s;
            return s;
        }
        CHECK(item->kernel);
        item->kernel_is_expensive = item->kernel->IsExpensive();
        item->kernel_is_async = (item->kernel->AsAsync() != nullptr);
        item->is_merge = IsMerge(n);
        item->is_enter = IsEnter(n);
        item->is_exit = IsExit(n);
        item->is_control_trigger = IsControlTrigger(n);
        item->is_sink = IsSink(n);
        item->is_enter_exit_or_next_iter = (IsEnter(n) || IsExit(n) || IsNextIteration(n));

        // Compute the maximum values we'll store for this node in the
        // pending counts data structure, and allocate a handle in
        // that frame's pending counts data structure that has enough
        // space to store these maximal count values.
        auto [max_pending, max_dead] = GetMaxPendingCounts(*n);
        item->pending_id = frame_info.pending_counts_layout.CreateHandle(max_pending, max_dead);

        // Initialize static information about the frames in the graph.
        frame_info.nodes.push_back(n);
        if (IsEnter(n)) {
            std::string enter_name;
            TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "frame_name", &enter_name));
            EnsureFrameInfo(enter_name).input_count++;
        }
    }

    // Initialize PendingCounts only after item->pending_id is initialized for
    // all nodes.
    InitializePending(*graph_, cf_info);

    return gview_.SetAllocAttrs(*graph_, params_.device);
}

Status ControlFlowInfo::Build(const tf::Graph &g)
{
    const auto num_nodes = g.num_node_ids();
    frame_names.resize(static_cast<size_t>(num_nodes));

    std::vector<tf::Node *> parent_nodes;
    parent_nodes.resize(static_cast<size_t>(num_nodes));
    std::vector<bool> visited;
    visited.resize(static_cast<size_t>(num_nodes));

    std::string frame_name;
    std::deque<tf::Node *> ready;

    // Initialize with the root nodes.
    for (auto n : g.nodes()) {
        if (n->in_edges().empty()) {
            visited[n->id()] = true;
            unique_frame_names.insert(frame_name);
            ready.push_back(n);
        }
    }

    while (!ready.empty()) {
        auto curr_node = ready.front();
        int curr_id = curr_node->id();
        ready.pop_front();

        tf::Node *parent = nullptr;
        if (IsEnter(curr_node)) {
            // Enter a child frame.
            TF_RETURN_IF_ERROR(GetNodeAttr(curr_node->def(), "frame_name", &frame_name));
            parent = curr_node;
        } else if (IsExit(curr_node)) {
            // Exit to the parent frame.
            parent = parent_nodes[curr_id];
            frame_name = frame_names[parent->id()];
            parent = parent_nodes[parent->id()];
        } else {
            parent = parent_nodes[curr_id];
            frame_name = frame_names[curr_id];
        }
        for (const auto *out_edge : curr_node->out_edges()) {
            auto out = out_edge->dst();
            int out_id = out->id();

            // Add to ready queue if not visited.
            bool is_visited = visited[out_id];
            if (!is_visited) {
                ready.push_back(out);
                visited[out_id] = true;

                // Process the node 'out'.
                frame_names[out_id] = frame_name;
                parent_nodes[out_id] = parent;
                unique_frame_names.insert(frame_name);
            }
        }
    }

    return Status::OK();
}

FrameInfo &SIExecutor::EnsureFrameInfo(const std::string &fname)
{
    return frame_info_[fname];
}

void SIExecutor::InitializePending(const tf::Graph &graph, const ControlFlowInfo &cf_info)
{
    for (auto &fname : cf_info.unique_frame_names) {
        auto &finfo = EnsureFrameInfo(fname);
        DCHECK_EQ(finfo.pending_counts, nullptr);
        finfo.pending_counts = std::make_unique<tf::PendingCounts>(finfo.pending_counts_layout);
    }

    for (const auto *n : graph.nodes()) {
        const int id = n->id();
        const auto &name = cf_info.frame_names[id];
        auto item = gview_.node(id);

        auto [max_pending, max_dead] = GetMaxPendingCounts(*n);
        UNUSED(max_dead);

        EnsureFrameInfo(name).pending_counts->set_initial_count(item->pending_id, max_pending);
    }
}

void SIExecutor::RunAsync(const tf::Executor::Args &args, std::function<void(const Status &)> done)
{
    params_.ins->scheduleIteartion(std::make_unique<SIExecutorTask>(*this, args, std::move(done)));
}

} // namespace salus::oplib::tensorflow
