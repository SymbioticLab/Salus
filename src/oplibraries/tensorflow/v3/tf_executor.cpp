//
// Created by peifeng on 7/24/18.
//

#include "oplibraries/tensorflow/v3/tf_executor.h"

#include "execution/engine/iterationcontext.h"
#include "execution/iterationtask.h"
#include "oplibraries/tensorflow/tfinstance.h"

namespace salus::oplib::tensorflow {

namespace {

// 1-D, 0 element tensor.
const tf::Tensor *const kEmptyTensor = new tf::Tensor;

bool IsInitializationOp(const tf::Node *node)
{
    return node->op_def().allows_uninitialized_input();
}

class ExecutorImpl;
class GraphView;

struct EdgeInfo
{
    int dst_id;
    int output_slot : 31;
    // true if this is the last info for output_slot in the EdgeInfo list.
    bool is_last : 1;
    int input_slot;
};

struct NodeItem
{
    NodeItem() {}

    // A graph node.
    const tf::Node *node = nullptr;

    // The kernel for this node.
    tf::OpKernel *kernel = nullptr;

    bool kernel_is_expensive : 1; // True iff kernel->IsExpensive()
    bool kernel_is_async : 1;     // True iff kernel->AsAsync() != nullptr
    bool is_merge : 1;            // True iff IsMerge(node)
    bool is_enter : 1;            // True iff IsEnter(node)
    bool is_exit : 1;             // True iff IsExit(node)
    bool is_control_trigger : 1;  // True iff IsControlTrigger(node)
    bool is_sink : 1;             // True iff IsSink(node)
    // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)
    bool is_enter_exit_or_next_iter : 1;

    // Cached values of node->num_inputs() and node->num_outputs(), to
    // avoid levels of indirection.
    int num_inputs;
    int num_outputs;

    // ExecutorImpl::tensors_[input_start] is the 1st positional input
    // for this node.
    int input_start = 0;

    // Number of output edges.
    size_t num_output_edges;

    tf::PendingCounts::Handle pending_id;

    const EdgeInfo *output_edge_list() const
    {
        return output_edge_base();
    }

    // ith output edge.
    const EdgeInfo &output_edge(int i) const
    {
        DCHECK_GE(i, 0);
        DCHECK_LT(static_cast<size_t>(i), num_output_edges);
        return output_edge_base()[i];
    }

    tf::DataType input_type(int i) const
    {
        DCHECK_LT(i, num_inputs);
        return static_cast<tf::DataType>(input_type_base()[i]);
    }
    tf::DataType output_type(int i) const
    {
        DCHECK_LT(i, num_outputs);
        return static_cast<tf::DataType>(output_type_base()[i]);
    }

    // Return array of per-output allocator attributes.
    const tf::AllocatorAttributes *output_attrs() const
    {
        return output_attr_base();
    }

private:
    friend class GraphView;

    // Variable length section starts immediately after *this
    // (uint8 is enough for DataType).
    //   EdgeInfo            out_edges[num_out_edges];
    //   AllocatorAttributes output_attr[num_outputs];
    //   uint8               input_type[num_inputs];
    //   uint8               output_type[num_outputs];

    // Return pointer to variable length section.
    char *var() const
    {
        return const_cast<char *>(reinterpret_cast<const char *>(this) + sizeof(NodeItem));
    }

    EdgeInfo *output_edge_base() const
    {
        return reinterpret_cast<EdgeInfo *>(var());
    }
    tf::AllocatorAttributes *output_attr_base() const
    {
        return reinterpret_cast<tf::AllocatorAttributes *>(var() + sizeof(EdgeInfo) * num_output_edges);
    }
    tf::uint8 *input_type_base() const
    {
        return reinterpret_cast<tf::uint8 *>(var() + sizeof(EdgeInfo) * num_output_edges
                                             + sizeof(tf::AllocatorAttributes) * num_outputs);
    }
    tf::uint8 *output_type_base() const
    {
        return reinterpret_cast<tf::uint8 *>(var() + sizeof(EdgeInfo) * num_output_edges
                                             + sizeof(tf::AllocatorAttributes) * num_outputs
                                             + sizeof(tf::uint8) * num_inputs);
    }

    TF_DISALLOW_COPY_AND_ASSIGN(NodeItem);
};

typedef tf::gtl::InlinedVector<tf::TensorValue, 4> TensorValueVec;
typedef tf::gtl::InlinedVector<tf::DeviceContext *, 4> DeviceContextVec;
typedef tf::gtl::InlinedVector<tf::AllocatorAttributes, 4> AllocatorAttributeVec;

// Immutable view of a Graph organized for efficient execution.
class GraphView
{
public:
    GraphView()
        : space_(nullptr)
    {
    }
    ~GraphView();

    void Initialize(const tf::Graph *g);
    Status SetAllocAttrs(const tf::Graph *g, const tf::Device *device);

    NodeItem *node(size_t id) const
    {
        DCHECK_LT(id, static_cast<size_t>(num_nodes_));
        tf::uint32 offset = node_offsets_[id];
        return ((offset == tf::kuint32max) ? nullptr
                                           : reinterpret_cast<NodeItem *>(space_ + node_offsets_[id]));
    }

private:
    char *InitializeNode(char *ptr, const tf::Node *n);
    size_t NodeItemBytes(const tf::Node *n);

    tf::int32 num_nodes_ = 0;
    tf::uint32 *node_offsets_ = nullptr; // array of size "graph_.num_node_ids()"
    // node_offsets_[id] holds the byte offset for node w/ "id" in space_

    char *space_; // NodeItem objects are allocated here

    TF_DISALLOW_COPY_AND_ASSIGN(GraphView);
};

class ExecutorImpl : public tf::Executor
{
public:
    ExecutorImpl(const TFExecutorParams &p, std::unique_ptr<const tf::Graph> &&g)
        : params_(p)
        , graph_(std::move(g))
        , gview_()
        , is_main_iter(false)
        , graph_id_(static_cast<uint64_t>(++NextSeq))
    {
        CHECK(p.create_kernel != nullptr);
        CHECK(p.delete_kernel != nullptr);
    }

    ~ExecutorImpl() override
    {
        for (int i = 0; i < graph_->num_node_ids(); i++) {
            NodeItem *item = gview_.node(i);
            if (item != nullptr) {
                params_.delete_kernel(item->kernel);
            }
        }
        for (auto fiter : frame_info_) {
            delete fiter.second;
        }
    }

    Status Initialize();

    // Process all Nodes in the current graph, attempting to infer the
    // memory allocation attributes to be used wherever they may allocate
    // a tensor buffer.
    Status SetAllocAttrs();

    void RunAsync(const Args &args, DoneCallback done) override;

private:
    friend class ExecutorState;
    friend class TFExecutorTask;

    struct ControlFlowInfo
    {
        tf::gtl::FlatSet<tf::string> unique_frame_names;
        std::vector<tf::string> frame_names;
    };

    struct FrameInfo
    {
        FrameInfo()
            : input_count(0)
            , total_inputs(0)
            , pending_counts(nullptr)
            , nodes(nullptr)
        {
        }

        // The total number of inputs to a frame.
        int input_count;

        // The total number of input tensors of a frame.
        // == sum(nodes[*].num_inputs()) where nodes are the nodes in the frame.
        int total_inputs;

        // Used to determine the next place to allocate space in the
        // pending_counts data structure we'll eventually construct
        tf::PendingCounts::Layout pending_counts_layout;

        // Each frame has its own PendingCounts only for the nodes in the frame.
        tf::PendingCounts *pending_counts; // Owned

        // The nodes in a frame. Used only for debugging.
        std::vector<const tf::Node *> *nodes; // Owned

        ~FrameInfo()
        {
            delete pending_counts;
            delete nodes;
        }
    };

    static Status BuildControlFlowInfo(const tf::Graph *graph, ControlFlowInfo *cf_info);
    void InitializePending(const tf::Graph *graph, const ControlFlowInfo &cf_info);

    FrameInfo *EnsureFrameInfo(const tf::string &fname)
    {
        auto slot = &frame_info_[fname];
        if (*slot == nullptr) {
            *slot = new FrameInfo;
        }
        return *slot;
    }

    // Owned.
    TFExecutorParams params_;
    std::unique_ptr<const tf::Graph> graph_;
    GraphView gview_;

    // A cached value of params_
    bool device_record_tensor_accesses_ = false;

    // Root nodes (with no in edges) that should form the initial ready queue
    std::vector<const tf::Node *> root_nodes_;

    // Mapping from frame name to static information about the frame.
    // TODO(yuanbyu): We could cache it along with the graph so to avoid
    // the overhead of constructing it for each executor instance.
    tf::gtl::FlatMap<tf::string, FrameInfo *> frame_info_;

    bool is_main_iter;
    // a combination of graphHandle and partition
    const uint64_t graph_id_;

    static std::atomic_int_fast64_t NextSeq;

    TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
};

std::atomic_int_fast64_t ExecutorImpl::NextSeq{0};

// Infer memory allocation attributes of a node n's output,
// based on its use node dst.  Note that dst might not be directly
// connected to n by a single edge, but might be a downstream
// consumer of n's output by reference.  *attr is updated with any
// necessary attributes.
Status InferAllocAttr(const tf::Node *n, const tf::Node *dst,
                      const tf::DeviceNameUtils::ParsedName &local_dev_name, tf::AllocatorAttributes *attr);

GraphView::~GraphView()
{
    static_assert(std::is_trivially_destructible<tf::AllocatorAttributes>::value,
                  "Update code if AllocatorAttributes gains a destructor");
    static_assert(std::is_trivially_destructible<EdgeInfo>::value,
                  "Update code if EdgeInfo gains a destructor");
    for (int i = 0; i < num_nodes_; i++) {
        NodeItem *n = node(i);
        if (n != nullptr) {
            n->NodeItem::~NodeItem();
            // Memory for "n" itself is held in space_ & gets cleaned up below
        }
    }
    delete[] node_offsets_;
    delete[] space_;
}

size_t GraphView::NodeItemBytes(const tf::Node *n)
{
    const size_t num_output_edges = n->out_edges().size();
    const int num_inputs = n->num_inputs();
    const int num_outputs = n->num_outputs();

    // Compute number of bytes needed for NodeItem and variable length data.
    // We do not subtract sizeof(var) since num_inputs/num_outputs might
    // both be zero.
    const size_t raw_bytes = sizeof(NodeItem)                                // Fixed
                             + num_output_edges * sizeof(EdgeInfo)           // output_edges[...]
                             + num_outputs * sizeof(tf::AllocatorAttributes) // output_attr[...]
                             + num_inputs * sizeof(tf::uint8)                // input_type[num_inputs]
                             + num_outputs * sizeof(tf::uint8);              // output_type[num_outputs]
    static constexpr size_t kItemAlignment = sizeof(NodeItem *);
    static_assert(kItemAlignment % alignof(NodeItem) == 0, "NodeItem must be aligned with kItemAlignment");
    static_assert(kItemAlignment % alignof(EdgeInfo) == 0, "EdgeInfo must be aligned with kItemAlignment");
    static_assert(kItemAlignment % alignof(tf::AllocatorAttributes) == 0,
                  "AllocatorAttributes must be aligned with kItemAlignment");
    static_assert(sizeof(NodeItem) % alignof(EdgeInfo) == 0, "NodeItem must be aligned with EdgeInfo");
    static_assert(sizeof(NodeItem) % alignof(tf::AllocatorAttributes) == 0,
                  "NodeItem must be aligned with AllocatorAttributes");
    static_assert(sizeof(EdgeInfo) % alignof(tf::AllocatorAttributes) == 0,
                  "EdgeInfo must be aligned with AllocatorAttributes");
    const size_t bytes = ((raw_bytes + kItemAlignment - 1) / kItemAlignment) * kItemAlignment;
    return bytes;
}

char *GraphView::InitializeNode(char *ptr, const tf::Node *n)
{
    const int id = n->id();
    CHECK(node_offsets_[id] == tf::kuint32max); // Initial value in constructor

    const size_t bytes = NodeItemBytes(n);
    constexpr size_t kItemAlignment = sizeof(NodeItem *);
    CHECK_EQ(reinterpret_cast<uintptr_t>(ptr) % kItemAlignment, 0);
    NodeItem *item = reinterpret_cast<NodeItem *>(ptr);

    // We store a 32-bit offset relative to the beginning of space_, so that we
    // only need an array of 32-bit values to map from node id to the NodeItem*,
    // (versus 64 bits on most machines if we just stored an array of NodeItem*
    // pointers). Casting to int64 is needed on 32bit CPU to avoid comparing
    // values as "int" vs "size_t" in CHECK_LE.
    CHECK_LE(static_cast<tf::int64>(ptr - space_), tf::kuint32max);
    const auto offset = static_cast<tf::uint32>(ptr - space_);
    node_offsets_[id] = offset;
    ptr += bytes;

    const size_t num_output_edges = n->out_edges().size();
    const int num_inputs = n->num_inputs();
    const int num_outputs = n->num_outputs();

    new (item) NodeItem();
    item->num_inputs = num_inputs;
    item->num_outputs = num_outputs;
    item->num_output_edges = num_output_edges;

    // Fill output edges.
    // Keep track of the last EdgeInfo in the EdgeInfo array that references
    // a given output slot.  For all but the last, we need to do a copy of the
    // Tensor when propagating results downstream in the graph, but for the
    // last one, we can just do a move of the Tensor object to propagate it.
    tf::gtl::InlinedVector<EdgeInfo *, 4> last_indices(num_outputs, nullptr);
    EdgeInfo *dst_edge = item->output_edge_base();
    for (auto e : n->out_edges()) {
        dst_edge->dst_id = e->dst()->id();
        CHECK_LE(e->src_output(), 0x3FFFFFFF); // Must fit in 31 bits
        dst_edge->output_slot = e->src_output();
        dst_edge->is_last = false;
        const int output_slot = dst_edge->output_slot;
        if (output_slot >= 0) {
            last_indices[output_slot] = dst_edge;
        }
        dst_edge->input_slot = e->dst_input();
        dst_edge++;
    }
    for (EdgeInfo *edge_info : last_indices) {
        if (edge_info != nullptr) {
            edge_info->is_last = true;
        }
    }

    auto *output_attrs = item->output_attr_base();
    for (int i = 0; i < num_outputs; i++) {
        new (&output_attrs[i]) tf::AllocatorAttributes();
    }

    DCHECK_LT(tf::DataType_MAX, 255); // Must fit in uint8
    auto *input_types = item->input_type_base();
    for (int i = 0; i < num_inputs; i++) {
        input_types[i] = static_cast<tf::uint8>(n->input_type(i));
        DCHECK_EQ(item->input_type(i), n->input_type(i));
    }

    auto *output_types = item->output_type_base();
    for (int i = 0; i < num_outputs; i++) {
        output_types[i] = static_cast<tf::uint8>(n->output_type(i));
        DCHECK_EQ(item->output_type(i), n->output_type(i));
    }
    return ptr;
}

void GraphView::Initialize(const tf::Graph *g)
{
    CHECK(node_offsets_ == nullptr);
    const int num_nodes = g->num_node_ids();
    num_nodes_ = num_nodes;
    size_t total_bytes = 0;
    for (const auto *n : g->nodes()) {
        total_bytes += NodeItemBytes(n);
    }

    node_offsets_ = new tf::uint32[num_nodes];
    for (int i = 0; i < num_nodes; i++) {
        node_offsets_[i] = tf::kuint32max;
    }

    space_ = new char[total_bytes]; // NodeItem objects are allocated here
    char *ptr = space_;
    for (const auto *n : g->nodes()) {
        ptr = InitializeNode(ptr, n);
    }
    CHECK_EQ(ptr, space_ + total_bytes);
}

void GetMaxPendingCounts(const tf::Node *n, size_t *max_pending, size_t *max_dead_count)
{
    const size_t num_in_edges = n->in_edges().size();
    size_t initial_count;
    if (IsMerge(n)) {
        // merge waits all control inputs so we initialize the pending
        // count to be the number of control edges.
        tf::int32 num_control_edges = 0;
        for (const auto *edge : n->in_edges()) {
            if (edge->IsControlEdge()) {
                num_control_edges++;
            }
        }
        // Use bit 0 to indicate if we are waiting for a ready live data input.
        initial_count = 1 + (num_control_edges << 1);
    } else {
        initial_count = num_in_edges;
    }

    *max_pending = initial_count;
    *max_dead_count = num_in_edges;
}

Status ExecutorImpl::Initialize()
{
    gview_.Initialize(graph_.get());

    // Build the information about frames in this subgraph.
    ControlFlowInfo cf_info;
    TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph_.get(), &cf_info));

    // Cache this value so we make this virtual function call once, rather
    // that O(# steps * # nodes per step) times.
    device_record_tensor_accesses_ = params_.device->RequiresRecordingAccessedTensors();

    for (auto &it : cf_info.unique_frame_names) {
        EnsureFrameInfo(it)->nodes = new std::vector<const tf::Node *>;
    }

    // Preprocess every node in the graph to create an instance of op
    // kernel for each node.
    for (const auto *n : graph_->nodes()) {
        const int id = n->id();
        const auto &frame_name = cf_info.frame_names[id];
        FrameInfo *frame_info = EnsureFrameInfo(frame_name);

        // Check if this is main iteration
        if (n->name() == "salus_main_iter") {
            VLOG(2) << params_.session << ":" << params_.graphHandle << " is main iteration";
            is_main_iter = true;
        }

        // See if this node is a root node, and if so, add to root_nodes_.
        if (n->in_edges().empty()) {
            root_nodes_.push_back(n);
        }

        NodeItem *item = gview_.node(id);
        item->node = n;

        item->input_start = frame_info->total_inputs;
        frame_info->total_inputs += n->num_inputs();

        Status s = params_.create_kernel(n->def(), &item->kernel);
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
        size_t max_pending, max_dead;
        GetMaxPendingCounts(n, &max_pending, &max_dead);
        item->pending_id = frame_info->pending_counts_layout.CreateHandle(max_pending, max_dead);

        // Initialize static information about the frames in the graph.
        frame_info->nodes->push_back(n);
        if (IsEnter(n)) {
            tf::string enter_name;
            TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "frame_name", &enter_name));
            EnsureFrameInfo(enter_name)->input_count++;
        }
    }

    // Initialize PendingCounts only after item->pending_id is initialized for
    // all nodes.
    InitializePending(graph_.get(), cf_info);

    return gview_.SetAllocAttrs(graph_.get(), params_.device);
}

Status GraphView::SetAllocAttrs(const tf::Graph *g, const tf::Device *device)
{
    Status s;
    auto local_dev_name = device->parsed_name();

    for (const auto *n : g->nodes()) {
        NodeItem *item = node(n->id());
        auto *attrs = item->output_attr_base();

        // Examine the out edges of each node looking for special use
        // cases that may affect memory allocation attributes.
        for (auto e : n->out_edges()) {
            if (!e->IsControlEdge()) {
                tf::AllocatorAttributes attr;
                s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
                if (!s.ok())
                    return s;
                if (attr.value != 0) {
                    attrs[e->src_output()].Merge(attr);
                }
            }
        }

        for (int out = 0; out < n->num_outputs(); out++) {
            const auto *op_kernel = item->kernel;
            DCHECK_LT(static_cast<size_t>(out), op_kernel->output_memory_types().size());
            bool on_host = op_kernel->output_memory_types()[out] == tf::HOST_MEMORY;
            if (on_host) {
                tf::AllocatorAttributes h;
                h.set_on_host(on_host);
                attrs[out].Merge(h);
            }
        }
    }
    return s;
}

Status InferAllocAttr(const tf::Node *n, const tf::Node *dst,
                      const tf::DeviceNameUtils::ParsedName &local_dev_name, tf::AllocatorAttributes *attr)
{
    Status s;
    // Note that it's possible for *n to be a Recv and *dst to be a Send,
    // so these two cases are not mutually exclusive.
    if (IsRecv(n)) {
        tf::string src_name;
        s = GetNodeAttr(n->attrs(), "send_device", &src_name);
        if (!s.ok())
            return s;
        tf::DeviceNameUtils::ParsedName parsed_src_name;
        if (!tf::DeviceNameUtils::ParseFullName(src_name, &parsed_src_name)) {
            s = tf::errors::Internal("Bad send_device attr '", src_name, "' in node ", n->name());
            return s;
        }
        if (!tf::DeviceNameUtils::IsSameAddressSpace(parsed_src_name, local_dev_name)) {
            // Value is going to be the sink of an RPC.
            attr->set_nic_compatible(true);
            VLOG(2) << "node " << n->name() << " is the sink of an RPC in";
        } else if ((local_dev_name.type == "CPU" || n->IsHostRecv()) && parsed_src_name.type != "CPU") {
            // Value is going to be the sink of a local DMA from GPU to CPU (or other
            // types of accelerators).
            attr->set_gpu_compatible(true);
            VLOG(2) << "node " << n->name() << " is the sink of a gpu->cpu copy";
        } else {
            VLOG(2) << "default alloc case local type " << local_dev_name.type << " remote type "
                    << parsed_src_name.type;
        }
    }
    if (IsSend(dst)) {
        tf::string dst_name;
        s = GetNodeAttr(dst->attrs(), "recv_device", &dst_name);
        if (!s.ok())
            return s;
        tf::DeviceNameUtils::ParsedName parsed_dst_name;
        if (!tf::DeviceNameUtils::ParseFullName(dst_name, &parsed_dst_name)) {
            s = tf::errors::Internal("Bad recv_device attr '", dst_name, "' in node ", n->name());
            return s;
        }
        if (!tf::DeviceNameUtils::IsSameAddressSpace(parsed_dst_name, local_dev_name)) {
            // Value is going to be the source of an RPC.
            attr->set_nic_compatible(true);
            VLOG(2) << "node " << n->name() << " is the source of an RPC out";
        } else if ((local_dev_name.type == "CPU" || dst->IsHostSend()) && parsed_dst_name.type != "CPU") {
            // Value is going to be the source of a local DMA from CPU to GPU (or
            // other types of accelerators).
            // Note that this does not cover the case where the allocation of the
            // output tensor is not generated by the src: n.
            attr->set_gpu_compatible(true);
            VLOG(2) << "node " << n->name() << " is the source of a cpu->gpu copy";
        } else {
            VLOG(2) << "default alloc case local type " << local_dev_name.type << " remote type "
                    << parsed_dst_name.type;
        }
    }
    return s;
}

// The state associated with one invocation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class ExecutorState
{
public:
    ExecutorState(const tf::Executor::Args &args, ExecutorImpl *impl, tf::Executor::DoneCallback done);
    ~ExecutorState();

    void runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept;

private:
    // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
    // TODO(yuanbyu): A better way to do "has_value"?
    struct Entry
    {
        Entry() {}
        Entry(const Entry &other)
            : ref(other.ref)
            , ref_mu(other.ref_mu)
            , has_value(other.has_value)
            , val_field_is_set(other.val_field_is_set)
            , alloc_attr(other.alloc_attr)
            , device_context(other.device_context)
        {
            if (val_field_is_set) {
                val.Init(*other.val);
            }
        }
        ~Entry()
        {
            if (val_field_is_set)
                val.Destroy();
        }

        Entry &operator=(const Entry &other)
        {
            if (val_field_is_set) {
                val.Destroy();
            }
            ref = other.ref;
            ref_mu = other.ref_mu;
            has_value = other.has_value;
            val_field_is_set = other.val_field_is_set;
            alloc_attr = other.alloc_attr;
            device_context = other.device_context;
            if (val_field_is_set) {
                val.Init(*other.val);
            }
            return *this;
        }

        Entry &operator=(Entry &&other)
        {
            if (val_field_is_set) {
                val.Destroy();
            }
            ref = other.ref;
            ref_mu = other.ref_mu;
            has_value = other.has_value;
            val_field_is_set = other.val_field_is_set;
            alloc_attr = other.alloc_attr;
            device_context = other.device_context;
            if (val_field_is_set) {
                val.Init(std::move(*other.val));
            }
            return *this;
        }

        // Clears the <val> field.
        void ClearVal()
        {
            if (val_field_is_set) {
                val.Destroy();
                val_field_is_set = false;
                has_value = false;
            }
        }

        // A tensor value, if val_field_is_set.
        tf::ManualConstructor<tf::Tensor> val;

        tf::Tensor *ref = nullptr;   // A tensor reference.
        tf::mutex *ref_mu = nullptr; // mutex for *ref if ref is not nullptr.

        // Whether the value exists, either in <val> or <ref>.
        bool has_value = false;

        bool val_field_is_set = false;

        // The attributes of the allocator that creates the tensor.
        tf::AllocatorAttributes alloc_attr;

        // Every entry carries an optional DeviceContext containing
        // Device-specific information about how the Tensor was produced.
        tf::DeviceContext *device_context = nullptr;
    };

    // Contains a value for [node->id()] for the device context assigned by the
    // device at the beginning of a step.
    tf::DeviceContextMap device_context_map_;

    struct TaggedNode;
    typedef tf::gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
    typedef tf::gtl::InlinedVector<Entry, 4> EntryVector;

    struct IterationState
    {
        explicit IterationState(const tf::PendingCounts *pending_counts, int total_input_tensors)
            : input_tensors(new Entry[total_input_tensors])
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

        // The number of outstanding ops for each iteration.
        size_t outstanding_ops;

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
        explicit FrameState(const ExecutorImpl *impl, int parallel_iters)
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
        const ExecutorImpl *executor = nullptr;

        // The name of this frame, which is the concatenation of its parent
        // frame name, the iteration of the parent frame when this frame was
        // created, and the value of the attr 'frame_name'.
        tf::string frame_name;

        // The unique id for this frame. Generated by fingerprinting
        // frame_name.
        tf::uint64 frame_id;

        // The iteration id of its parent frame when this frame is created.
        // -1 if there is no parent frame. The frame_name/parent_iter pair
        // uniquely identifies this FrameState.
        tf::int64 parent_iter = -1;

        // The FrameState of its parent frame.
        FrameState *parent_frame = nullptr;

        // The maximum allowed number of parallel iterations.
        const int max_parallel_iterations;

        // The number of inputs this frame is still waiting.
        int num_pending_inputs = 0;

        // The highest iteration number we have reached so far in this frame.
        tf::int64 iteration_count GUARDED_BY(mu) = 0;

        // The number of outstanding iterations.
        int num_outstanding_iterations GUARDED_BY(mu) = 1;

        // The active iteration states of this frame.
        tf::gtl::InlinedVector<IterationState *, 12> iterations;

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
        tf::mutex mu;

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

        inline IterationState *GetIteration(tf::int64 iter) EXCLUSIVE_LOCKS_REQUIRED(mu)
        {
            size_t index = iter % iterations.size();
            return iterations[index];
        }

        inline void SetIteration(tf::int64 iter, IterationState *state) EXCLUSIVE_LOCKS_REQUIRED(mu)
        {
            size_t index = iter % iterations.size();
            DCHECK(state == nullptr || iterations[index] == nullptr);
            iterations[index] = state;
        }

        // Decrement the outstanding op count and clean up the iterations in the
        // frame. Return true iff the execution of the frame is done.
        inline bool DecrementOutstandingOps(const GraphView *gview, tf::int64 iter, TaggedNodeSeq *ready)
        {
            tf::mutex_lock l(mu);
            return DecrementOutstandingOpsLocked(gview, iter, ready);
        }

        // Decrement the outstanding op count and clean up the iterations in the
        // frame. Return true iff the execution of the frame is done.
        inline bool DecrementOutstandingOpsLocked(const GraphView *gview, tf::int64 iter,
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
        bool IsIterationDone(tf::int64 iter) EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Increments the iteration id. If this is a new iteration, initialize it.
        void IncrementIteration(const GraphView *gview, TaggedNodeSeq *ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Activate all the deferred NextIteration nodes in a new iteration.
        void ActivateNexts(const GraphView *gview, tf::int64 iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Activate all the current loop invariants in a new iteration.
        void ActivateLoopInvs(const GraphView *gview, tf::int64 iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Add a new loop invariant and make it available to all active iterations.
        void AddLoopInv(const NodeItem *item, const Entry &value, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Activate the successors of a node. Contents of *outputs are left in an
        // indeterminate state after returning from this method.
        void ActivateNodes(const NodeItem *item, const bool is_dead, tf::int64 iter, EntryVector *outputs,
                           TaggedNodeSeq *ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

        // Cleanup iterations of this frame starting from iteration iter.
        bool CleanupIterations(const GraphView *gview, tf::int64 iter, TaggedNodeSeq *ready)
            EXCLUSIVE_LOCKS_REQUIRED(mu);

        ~FrameState()
        {
            for (size_t i = 0; i < iterations.size(); ++i) {
                delete iterations[i];
                iterations[i] = nullptr;
            }
        }
    };

    // A tagged node: <frame*, iter, node*>.
    struct TaggedNode
    {
        const tf::Node *node = nullptr;
        FrameState *input_frame = nullptr;
        tf::int64 input_iter = -1;
        bool is_dead = false;

        TaggedNode(const tf::Node *t_node, FrameState *in_frame, tf::int64 in_iter, bool dead)
        {
            node = t_node;
            input_frame = in_frame;
            input_iter = in_iter;
            is_dead = dead;
        }
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
        tf::gtl::InlinedVector<TaggedNode, 16> ready_;
        size_t front_index_;
    };

    struct AsyncState;

    const bool vlog_; // true if VLOG_IS_ON(1). Used to check vlog cheaply.

    std::shared_ptr<IterationContext> ictx_;

    // true if LogMemory::IsEnabled(). Used to check memory enabled cheaply.
    const bool log_memory_;

    tf::int64 step_id_;
    // Not owned.
    tf::Rendezvous *rendezvous_;
    tf::SessionState *session_state_;
    tf::TensorStore *tensor_store_;
    // Step-local container.
    tf::ScopedStepContainer *step_container_;
    tf::StepStatsCollector *stats_collector_;
    // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
    // instead of a pointer?  (avoids having to delete).
    tf::checkpoint::TensorSliceReaderCacheWrapper *slice_reader_cache_;
    tf::CallFrameInterface *call_frame_;
    const ExecutorImpl *impl_;
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

    tf::mutex mu_;
    Status status_ GUARDED_BY(mu_);

    // Mapping from frame name to outstanding frames. A new frame is created
    // at some iteration of an active frame. So the unique key for the new
    // child frame is composed of the name of the parent frame, the iteration
    // number at which the parent frame is creating the new frame, and the
    // name of the new frame from nodedef.
    tf::gtl::FlatMap<tf::string, FrameState *> outstanding_frames_ GUARDED_BY(mu_);

    // The unique name of a frame.
    inline tf::string MakeFrameName(FrameState *frame, tf::int64 iter_id, const tf::string &name)
    {
        return tf::strings::StrCat(frame->frame_name, ";", iter_id, ";", name);
    }

    // Find an existing or create a new child frame in the frame 'frame' at
    // iteration 'iter'.
    void FindOrCreateChildFrame(FrameState *frame, tf::int64 iter, const tf::Node *node, FrameState **child);

    // Delete a frame. Called when the frame is done.
    void DeleteFrame(FrameState *frame, TaggedNodeSeq *ready);

    // Cleanup frames and iterations starting from frame/iter. Called when
    // a child frame is done.
    void CleanupFramesIterations(FrameState *frame, tf::int64 iter, TaggedNodeSeq *ready);

    // Process a ready node in current thread.
    void Process(TaggedNode node, tf::int64 scheduled_usec);

    // Before invoking item->kernel, fills in its "inputs".
    Status PrepareInputs(const NodeItem &item, Entry *first_input, TensorValueVec *inputs,
                         DeviceContextVec *input_device_contexts, AllocatorAttributeVec *input_alloc_attrs,
                         bool *is_input_dead);

    // After item->kernel computation is done, processes its outputs.
    Status ProcessOutputs(const NodeItem &item, tf::OpKernelContext *ctx, EntryVector *outputs,
                          tf::NodeExecStatsWrapper *stats);

    // After processing the outputs, propagates the outputs to their dsts.
    // Contents of *outputs are left in an indeterminate state after
    // returning from this method.
    void PropagateOutputs(const TaggedNode &tagged_node, const NodeItem *item, EntryVector *outputs,
                          TaggedNodeSeq *ready);

    // "node" just finishes. Takes ownership of "stats". Returns true if
    // execution has completed.
    bool NodeDone(const Status &s, const tf::Node *node, const TaggedNodeSeq &ready,
                  tf::NodeExecStatsWrapper *stats, TaggedNodeReadyQueue *inline_ready);

    // Schedule all the expensive nodes in 'ready', and put all the inexpensive
    // nodes in 'ready' into 'inline_ready'.
    void ScheduleReady(const TaggedNodeSeq &ready, TaggedNodeReadyQueue *inline_ready);

    // For debugging/logging only.
    inline void MaybeMarkCompleted(FrameState *frame, tf::int64 iter, tf::int64 id);

    // Clean up when this executor is done.
    void Finish();

    // A standalone routine for this expression so that we can express
    // that we don't want thread safety analysis on this reference (it's
    // safe to do without the lock because the iterations array never
    // resizes and this particular iteration's array element will not
    // be changed out from under us because the iteration is still alive).
    Entry *GetInputTensors(FrameState *input_frame, tf::int64 input_iter) const NO_THREAD_SAFETY_ANALYSIS
    {
        return input_frame->GetIteration(input_iter)->input_tensors;
    }
};

ExecutorState::ExecutorState(const tf::Executor::Args &args, ExecutorImpl *impl,
                             tf::Executor::DoneCallback done)
    : vlog_(VLOG_IS_ON(1))
    , log_memory_(tf::LogMemory::IsEnabled())
    , step_id_(args.step_id)
    , rendezvous_(args.rendezvous)
    , session_state_(args.session_state)
    , tensor_store_(args.tensor_store)
    , step_container_(args.step_container)
    , stats_collector_(args.stats_collector)
    , slice_reader_cache_(new tf::checkpoint::TensorSliceReaderCacheWrapper)
    , call_frame_(args.call_frame)
    , impl_(impl)
    , cancellation_manager_(args.cancellation_manager)
    , runner_(args.runner)
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
    root_frame_->iterations.resize(root_frame_->max_parallel_iterations);
    root_frame_->iterations[0] =
        new IterationState(root_frame_->pending_counts, root_frame_->total_input_tensors);

    outstanding_frames_.insert({root_frame_->frame_name, root_frame_});
}

ExecutorState::~ExecutorState()
{
    for (auto name_frame : outstanding_frames_) {
        delete name_frame.second;
    }
    for (auto it : device_context_map_) {
        it->Unref();
    }
    delete slice_reader_cache_;
}

Status ExecutorImpl::BuildControlFlowInfo(const tf::Graph *g, ControlFlowInfo *cf_info)
{
    const int num_nodes = g->num_node_ids();
    cf_info->frame_names.resize(num_nodes);
    std::vector<tf::Node *> parent_nodes;
    parent_nodes.resize(num_nodes);
    std::vector<bool> visited;
    visited.resize(num_nodes);

    tf::string frame_name;
    std::deque<tf::Node *> ready;

    // Initialize with the root nodes.
    for (auto *n : g->nodes()) {
        if (n->in_edges().empty()) {
            visited[n->id()] = true;
            cf_info->unique_frame_names.insert(frame_name);
            ready.push_back(n);
        }
    }

    while (!ready.empty()) {
        auto *curr_node = ready.front();
        int curr_id = curr_node->id();
        ready.pop_front();

        tf::Node *parent = nullptr;
        if (IsEnter(curr_node)) {
            // Enter a child frame.
            TF_RETURN_IF_ERROR(GetNodeAttr(curr_node->attrs(), "frame_name", &frame_name));
            parent = curr_node;
        } else if (IsExit(curr_node)) {
            // Exit to the parent frame.
            parent = parent_nodes[curr_id];
            frame_name = cf_info->frame_names[parent->id()];
            parent = parent_nodes[parent->id()];
        } else {
            parent = parent_nodes[curr_id];
            frame_name = cf_info->frame_names[curr_id];
        }

        for (const auto *out_edge : curr_node->out_edges()) {
            auto *out = out_edge->dst();
            const int out_id = out->id();

            // Add to ready queue if not visited.
            bool is_visited = visited[out_id];
            if (!is_visited) {
                ready.push_back(out);
                visited[out_id] = true;

                // Process the node 'out'.
                cf_info->frame_names[out_id] = frame_name;
                parent_nodes[out_id] = parent;
                cf_info->unique_frame_names.insert(frame_name);
            }
        }
    }

    return Status::OK();
}

void ExecutorImpl::InitializePending(const tf::Graph *graph, const ControlFlowInfo &cf_info)
{
    for (auto &it : cf_info.unique_frame_names) {
        FrameInfo *finfo = EnsureFrameInfo(it);
        auto *counts = new tf::PendingCounts(finfo->pending_counts_layout);
        DCHECK_EQ(finfo->pending_counts, nullptr);
        finfo->pending_counts = counts;
    }
    for (const auto *n : graph->nodes()) {
        const int id = n->id();
        const auto &name = cf_info.frame_names[id];
        size_t max_pending, max_dead;
        GetMaxPendingCounts(n, &max_pending, &max_dead);
        const NodeItem *item = gview_.node(id);
        auto *counts = EnsureFrameInfo(name)->pending_counts;
        counts->set_initial_count(item->pending_id, max_pending);
    }
}

void ExecutorState::runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept
{
    ictx_ = std::move(ictx);
    ictx_->setGraphId(impl_->graph_id_);

    LogOpTracing() << "event: start_iter "
                   << nlohmann::json({
                          {"sess", impl_->params_.session},
                          {"stepId", step_id_},
                          {"mainIter", impl_->is_main_iter},
                          {"graphId", impl_->graph_id_},
                      });

    const tf::Graph *graph = impl_->graph_.get();
    TaggedNodeSeq ready;

    // Ask the device to fill in the device context map.
    auto *device = impl_->params_.device;
    const Status fill_status = device->FillContextMap(graph, &device_context_map_);
    if (!fill_status.ok()) {
        done_cb_(fill_status);
        return;
    }

    // Initialize the ready queue.
    for (const auto *n : impl_->root_nodes_) {
        DCHECK_EQ(n->in_edges().size(), 0);
        LogOpTracing() << "event: queued "
                       << nlohmann::json({
                                             {"name", n->name()},
                                             {"type", n->type_string()},
                                             {"session", impl_->params_.session},
                                             {"graphId", impl_->graph_id_},
                                             {"mainIter", impl_->is_main_iter},
                                             {"stepId", step_id_},
                                         });
        ready.push_back(TaggedNode{n, root_frame_, 0, false});
    }
    if (ready.empty()) {
        done_cb_(Status::OK());
    } else {
        num_outstanding_ops_ = ready.size();
        root_frame_->iterations[0]->outstanding_ops = ready.size();
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
struct ExecutorState::AsyncState
{
    AsyncState(const tf::OpKernelContext::Params &p, const TaggedNode &_tagged_node, const NodeItem *_item,
               Entry *_first_input, tf::NodeExecStatsWrapper *_stats)
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
        , stats(_stats)
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
    tf::NodeExecStatsWrapper *stats;

private:
    tf::OpKernelContext::Params *ParamsButClearingEigenGPUDevice(tf::OpKernelContext::Params *p)
    {
        // Ensure OpKernelContext constructor will make a new eigen GPU device if
        // necessary.
        p->eigen_gpu_device = nullptr; // Force allocation
        return p;
    }
};

void ExecutorState::Process(TaggedNode tagged_node, tf::int64)
{
    const GraphView &gview = impl_->gview_;
    TaggedNodeSeq ready;
    TaggedNodeReadyQueue inline_ready;

    // Parameters passed to OpKernel::Compute.
    TensorValueVec inputs;
    DeviceContextVec input_device_contexts;
    AllocatorAttributeVec input_alloc_attrs;

    tf::OpKernelContext::Params params;
    params.step_id = step_id_;
    auto *device = impl_->params_.device;
    params.device = device;
    params.log_memory = log_memory_;
    params.record_tensor_accesses = impl_->device_record_tensor_accesses_;
    params.rendezvous = rendezvous_;
    params.session_state = session_state_;
    params.tensor_store = tensor_store_;
    params.cancellation_manager = cancellation_manager_;
    params.call_frame = call_frame_;
    params.function_library = impl_->params_.function_library;
    params.resource_manager = device->resource_manager();
    params.step_container = step_container_;
    params.slice_reader_cache = slice_reader_cache_;
    params.inputs = &inputs;
    params.input_device_contexts = &input_device_contexts;
    params.input_alloc_attrs = &input_alloc_attrs;
    params.runner = &runner_;
    params.stats_collector = stats_collector_;

    Status s;
    EntryVector outputs;
    bool completed = false;
    inline_ready.push_back(tagged_node);
    while (!inline_ready.empty()) {
        tagged_node = inline_ready.front();
        inline_ready.pop_front();
        const auto *node = tagged_node.node;
        FrameState *input_frame = tagged_node.input_frame;
        const tf::int64 input_iter = tagged_node.input_iter;
        const int id = node->id();
        const NodeItem &item = *gview.node(id);

        // TODO(misard) Replace with a finer-grain enabling flag once we
        // add better optional debugging support.
        if (vlog_ && VLOG_IS_ON(1)) {
            tf::mutex_lock l(input_frame->mu);
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
        LogOpTracing() << "event: running "
                       << nlohmann::json({
                                             {"name", tagged_node.node->name()},
                                             {"type", tagged_node.node->type_string()},
                                             {"session", impl_->params_.session},
                                             {"graphId", impl_->graph_id_},
                                             {"mainIter", impl_->is_main_iter},
                                             {"stepId", step_id_},
                                         });

        Entry *input_tensors = GetInputTensors(input_frame, input_iter);
        Entry *first_input = input_tensors + item.input_start;
        outputs.clear();

        tf::TensorReferenceVector accessed_tensors;
        tf::DeviceContext *device_context = nullptr;
        // Only execute this node if it is not dead or it is a send/recv
        // transfer node. For transfer nodes, we need to propagate the "dead"
        // bit even when the node is dead.
        bool launched_asynchronously = false;
        if (tagged_node.is_dead && !IsTransferNode(node)) {
            outputs.resize(item.num_outputs);
        } else {
            // Prepares inputs.
            bool is_input_dead = false;
            s = PrepareInputs(item, first_input, &inputs, &input_device_contexts, &input_alloc_attrs,
                              &is_input_dead);
            if (!s.ok()) {
                // Clear inputs.
                int num_inputs = item.num_inputs;
                for (int i = 0; i < num_inputs; ++i) {
                    (first_input + i)->ClearVal();
                }
                MaybeMarkCompleted(input_frame, input_iter, id);
                // Continue to process the nodes in 'inline_ready'.
                completed = NodeDone(s, item.node, ready, nullptr, &inline_ready);
                continue;
            }

            // Set up compute params.
            auto *op_kernel = item.kernel;
            params.op_kernel = op_kernel;
            params.frame_iter = tf::FrameAndIter(input_frame->frame_id, input_iter);
            params.is_input_dead = is_input_dead;
            params.output_attr_array = item.output_attrs();

            if (item.kernel_is_async) {
                // Asynchronous computes.
                auto *async = item.kernel->AsAsync();
                DCHECK(async != nullptr);
                launched_asynchronously = true;
                AsyncState *state = new AsyncState(params, tagged_node, &item, first_input, nullptr);

                auto done = [this, state]() {
                    auto *device = impl_->params_.device;
                    Entry *first_input = state->first_input; // Shorthand

                    EntryVector outputs;
                    Status s = ProcessOutputs(*state->item, &state->ctx, &outputs, nullptr);
                    if (vlog_) {
                        VLOG(2) << "Async kernel done: " << state->item->node->id() << " step " << step_id_
                                << " " << SummarizeNode(*state->item->node)
                                << " is dead: " << state->tagged_node.is_dead;
                    }
                    // Clears inputs.
                    const int num_inputs = state->item->num_inputs;
                    for (int i = 0; i < num_inputs; ++i) {
                        (first_input + i)->ClearVal();
                    }
                    FrameState *input_frame = state->tagged_node.input_frame;
                    const tf::int64 input_iter = state->tagged_node.input_iter;
                    const int id = state->tagged_node.node->id();
                    MaybeMarkCompleted(input_frame, input_iter, id);
                    TaggedNodeSeq ready;
                    if (s.ok()) {
                        PropagateOutputs(state->tagged_node, state->item, &outputs, &ready);
                    }
                    outputs.clear();
                    if (s.ok() && impl_->device_record_tensor_accesses_) {
                        // Get the list of all tensors accessed during the execution
                        tf::TensorReferenceVector accessed;
                        state->ctx.retrieve_accessed_tensors(&accessed);
                        // callee takes ownership of the vector
                        device->ConsumeListOfAccessedTensors(state->ctx.op_device_context(), accessed);
                    }
                    const bool completed = NodeDone(s, state->item->node, ready, nullptr, nullptr);
                    delete state;
                    if (completed)
                        Finish();
                };
                device->ComputeAsync(async, &state->ctx, done);
            } else {
                // Synchronous computes.
                tf::OpKernelContext ctx(&params, item.num_outputs);
                CHECK_NOTNULL(op_kernel);
                device->Compute(op_kernel, &ctx);
                s = ProcessOutputs(item, &ctx, &outputs, nullptr);
                if (s.ok() && impl_->device_record_tensor_accesses_) {
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
                (first_input + i)->ClearVal();
            }
            MaybeMarkCompleted(input_frame, input_iter, id);
            // Propagates outputs.
            if (s.ok()) {
                PropagateOutputs(tagged_node, &item, &outputs, &ready);
            }
            outputs.clear();
            if (!accessed_tensors.empty()) {
                // device_context is set above in synchronous computes
                device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
            }
            // Postprocess.
            completed = NodeDone(s, item.node, ready, nullptr, &inline_ready);
        }
    } // while !inline_ready.empty()

    // This thread of computation is done if completed = true.
    if (completed)
        Finish();
}

Status ExecutorState::PrepareInputs(const NodeItem &item, Entry *first_input, TensorValueVec *inputs,
                                    DeviceContextVec *input_device_contexts,
                                    AllocatorAttributeVec *input_alloc_attrs, bool *is_input_dead)
{
    const auto *node = item.node;

    inputs->clear();
    inputs->resize(item.num_inputs);
    input_device_contexts->clear();
    input_device_contexts->resize(item.num_inputs);
    input_alloc_attrs->clear();
    input_alloc_attrs->resize(item.num_inputs);

    *is_input_dead = false;

    bool is_merge = item.is_merge;
    for (int i = 0; i < item.num_inputs; ++i) {
        const bool expect_ref = IsRefType(item.input_type(i));
        Entry *entry = first_input + i;
        (*input_device_contexts)[i] = entry->device_context;
        (*input_alloc_attrs)[i] = entry->alloc_attr;

        // i-th input.
        auto *inp = &(*inputs)[i];

        // Only merge and transfer nodes can have no-value inputs.
        if (!entry->has_value) {
            if (!is_merge) {
                DCHECK(IsTransferNode(node)) << node->name() << " - input " << i;
                DCHECK(!entry->val_field_is_set) << node->name() << " - input " << i;
                entry->has_value = true;
                entry->val_field_is_set = true;
                entry->val.Init(*kEmptyTensor);
                inp->tensor = entry->val.get();
                *is_input_dead = true;
            }
            continue;
        }
        if (entry->ref == nullptr) {
            if (expect_ref) {
                return AttachDef(tf::errors::InvalidArgument(i, "-th input expects a ref type"),
                                 item.kernel->def());
            }
            inp->tensor = entry->val.get();
        } else {
            {
                tf::mutex_lock ml(*entry->ref_mu);
                if (!entry->ref->IsInitialized() && !IsInitializationOp(item.node)) {
                    return AttachDef(tf::errors::FailedPrecondition("Attempting to use uninitialized value ",
                                                                    item.kernel->requested_input(i)),
                                     item.kernel->def());
                }
            }
            if (expect_ref) {
                inp->mutex_if_ref = entry->ref_mu;
                inp->tensor = entry->ref;
            } else {
                // Automatically deref the tensor ref when the op expects a
                // tensor but is given a ref to a tensor.  Need to deref it
                // under the mutex.
                {
                    tf::mutex_lock l(*(entry->ref_mu));
                    DCHECK(!entry->val_field_is_set);
                    entry->val.Init(*entry->ref);
                    entry->val_field_is_set = true;
                }
                entry->ref = nullptr;
                entry->ref_mu = nullptr;

                inp->tensor = entry->val.get();
            }
        }
    }
    return Status::OK();
}

Status ExecutorState::ProcessOutputs(const NodeItem &item, tf::OpKernelContext *ctx, EntryVector *outputs,
                                     tf::NodeExecStatsWrapper *)
{
    const auto *node = item.node;
    DCHECK_EQ(0, outputs->size());
    outputs->resize(item.num_outputs);

    Status s = ctx->status();
    if (!s.ok()) {
        s = AttachDef(s, item.kernel->def());
        // TODO(misard) Replace with a finer-grain enabling flag once we
        // add better optional debugging support.
        if (vlog_ && VLOG_IS_ON(1)) {
            LOG(WARNING) << this << " Compute status: " << s;
        }
        if (s.code() == tf::error::RESOURCE_EXHAUSTED) {
            if (stats_collector_) {
                auto err = stats_collector_->ReportAllocsOnResourceExhausted(s.error_message());
                s = Status(s.code(), tf::strings::StrCat(s.error_message(), err));
            } else {
                s = Status(s.code(),
                           tf::strings::StrCat(s.error_message(),
                                               "\nHint: If you want to see a list of allocated tensors when "
                                               "OOM happens, add report_tensor_allocations_upon_oom "
                                               "to RunOptions for current allocation info.\n"));
            }
        }
        return s;
    }

    // Get the device_context for this node id, if it exists.
    tf::DeviceContext *device_context = nullptr;
    if (static_cast<size_t>(node->id()) < device_context_map_.size()) {
        device_context = device_context_map_[node->id()];
    }

    // Experimental: debugger (tfdb) access to intermediate node completion.
    if (item.num_outputs == 0 && impl_->params_.node_outputs_cb != nullptr) {
        // If the node has no output, invoke the callback with output slot set to
        // -1, signifying that this is a no-output node.
        s.Update(impl_->params_.node_outputs_cb(item.node->name(), -1, nullptr, false, ctx));
    }

    for (int i = 0; i < item.num_outputs; ++i) {
        const tf::TensorValue val = ctx->release_output(i);
        if (*ctx->is_output_dead() || val.tensor == nullptr) {
            // Unless it's a Switch or a Recv, the node must produce a
            // tensor value at i-th output.
            if (!IsSwitch(node) && !IsRecv(node)) {
                s.Update(tf::errors::Internal("Missing ", i, "-th output from ", SummarizeNode(*node)));
            }
        } else {
            Entry *out = &((*outputs)[i]);

            // Set the device context of the output entry.
            out->device_context = device_context;

            // Set the allocator attributes of the output entry.
            out->alloc_attr = ctx->output_alloc_attr(i);

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
                    out->has_value = true;
                    out->ref = val.tensor;
                    out->ref_mu = val.mutex_if_ref;

                    // Experimental: debugger (tfdb) access to intermediate node
                    // outputs.
                    if (impl_->params_.node_outputs_cb != nullptr) {
                        s.Update(impl_->params_.node_outputs_cb(item.node->name(), i, out->ref, true, ctx));
                    }
                } else {
                    // NOTE that std::move is used here, so val.tensor goes to
                    // uninitialized state (val.tensor->IsInitialized return false).
                    DCHECK(!out->val_field_is_set);
                    out->has_value = true;
                    out->val_field_is_set = true;
                    out->val.Init(std::move(*val.tensor));

                    // Experimental: debugger access to intermediate node outputs.
                    if (impl_->params_.node_outputs_cb != nullptr) {
                        s.Update(
                            impl_->params_.node_outputs_cb(item.node->name(), i, out->val.get(), false, ctx));
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

void ExecutorState::PropagateOutputs(const TaggedNode &tagged_node, const NodeItem *item,
                                     EntryVector *outputs, TaggedNodeSeq *ready)
{
    const auto *node = tagged_node.node;
    FrameState *input_frame = tagged_node.input_frame;
    const tf::int64 input_iter = tagged_node.input_iter;
    const bool is_dead = tagged_node.is_dead;

    // Propagates outputs along out edges, and puts newly ready nodes
    // into the ready queue.
    ready->clear();
    bool is_frame_done = false;
    FrameState *output_frame = input_frame;
    tf::int64 output_iter = input_iter;

    if (!item->is_enter_exit_or_next_iter) {
        // Fast path for nodes types that don't need special handling
        DCHECK_EQ(input_frame, output_frame);
        // Normal path for most nodes
        tf::mutex_lock l(input_frame->mu);
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
        is_frame_done = input_frame->DecrementOutstandingOpsLocked(&impl_->gview_, input_iter, ready);
    } else if (item->is_enter) {
        bool is_constant;
        const Status s = GetNodeAttr(node->attrs(), "is_constant", &is_constant);
        DCHECK(s.ok()) << s;
        FindOrCreateChildFrame(input_frame, input_iter, node, &output_frame);
        output_iter = 0;
        {
            const NodeItem *item = impl_->gview_.node(node->id());
            tf::mutex_lock l(output_frame->mu);
            if (is_constant) {
                // Propagate to all active iterations if this is a loop invariant.
                output_frame->AddLoopInv(item, (*outputs)[0], ready);
            } else {
                output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
            }
            output_frame->num_pending_inputs--;
        }
        is_frame_done = input_frame->DecrementOutstandingOps(&impl_->gview_, input_iter, ready);
    } else if (item->is_exit) {
        if (is_dead) {
            tf::mutex_lock l(input_frame->mu);
            // Stop and remember this node if it is a dead exit.
            if (input_iter == input_frame->iteration_count) {
                input_frame->dead_exits.push_back(node);
            }
            is_frame_done = input_frame->DecrementOutstandingOpsLocked(&impl_->gview_, input_iter, ready);
        } else {
            output_frame = input_frame->parent_frame;
            output_iter = input_frame->parent_iter;
            {
                tf::mutex_lock l(output_frame->mu);
                output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
            }
            is_frame_done = input_frame->DecrementOutstandingOps(&impl_->gview_, input_iter, ready);
        }
    } else {
        DCHECK(IsNextIteration(node));
        tf::mutex_lock l(input_frame->mu);
        if (is_dead) {
            // Stop the deadness propagation.
            output_frame = nullptr;
        } else {
            if (input_iter == input_frame->iteration_count
                && input_frame->num_outstanding_iterations == input_frame->max_parallel_iterations) {
                // Reached the maximum for parallel iterations.
                input_frame->next_iter_roots.push_back({node, (*outputs)[0]});
                output_frame = nullptr;
            } else {
                // If this is a new iteration, start it.
                if (input_iter == input_frame->iteration_count) {
                    input_frame->IncrementIteration(&impl_->gview_, ready);
                }
                output_iter = input_iter + 1;
            }
        }
        if (output_frame != nullptr) {
            // This is the case when node is not Enter, Exit, or NextIteration.
            DCHECK(input_frame == output_frame);
            output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
        }
        is_frame_done = input_frame->DecrementOutstandingOpsLocked(&impl_->gview_, input_iter, ready);
    }

    for (const auto &n : *ready) {
        LogOpTracing() << "event: queued "
                       << nlohmann::json({
                                             {"name", n.node->name()},
                                             {"type", n.node->type_string()},
                                             {"session", impl_->params_.session},
                                             {"graphId", impl_->graph_id_},
                                             {"mainIter", impl_->is_main_iter},
                                             {"stepId", step_id_},
                                         });
    }


    // At this point, this node is completely done. We also know if the
    // completion of this node makes its frame completed.
    if (is_frame_done) {
        FrameState *parent_frame = input_frame->parent_frame;
        const tf::int64 parent_iter = input_frame->parent_iter;
        DeleteFrame(input_frame, ready);
        if (parent_frame != nullptr) {
            // The completion of frame may cause completions in its parent frame.
            // So clean things up recursively.
            CleanupFramesIterations(parent_frame, parent_iter, ready);
        }
    }
}

bool ExecutorState::NodeDone(const Status &s, const tf::Node *node, const TaggedNodeSeq &ready,
                             tf::NodeExecStatsWrapper *, TaggedNodeReadyQueue *inline_ready)
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
        tf::mutex_lock l(mu_);
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
    const size_t ready_size = ready.size();
    if (ready_size == 0 || !s.ok()) {
        completed = (num_outstanding_ops_.fetch_sub(1) == 1);
    } else if (ready_size > 1) {
        num_outstanding_ops_.fetch_add(ready_size - 1, std::memory_order_relaxed);
    }

    LogOpTracing() << "event: done "
                   << nlohmann::json({
                                         {"name", node->name()},
                                         {"type", node->type_string()},
                                         {"session", impl_->params_.session},
                                         {"graphId", impl_->graph_id_},
                                         {"mainIter", impl_->is_main_iter},
                                         {"stepId", step_id_},
                                     });

    // Schedule the ready nodes in 'ready'.
    if (s.ok()) {
        ScheduleReady(ready, inline_ready);
    }
    return completed;
}

void ExecutorState::ScheduleReady(const TaggedNodeSeq &ready, TaggedNodeReadyQueue *inline_ready)
{
    if (ready.empty())
        return;

    tf::int64 scheduled_usec = 0;
    if (inline_ready == nullptr) {
        // Schedule to run all the ready ops in thread pool.
        for (auto &tagged_node : ready) {
            runner_([=]() { Process(tagged_node, scheduled_usec); });
        }
        return;
    }
    const GraphView &gview = impl_->gview_;
    const TaggedNode *curr_expensive_node = nullptr;
    for (auto &tagged_node : ready) {
        const NodeItem &item = *gview.node(tagged_node.node->id());
        if (tagged_node.is_dead || !item.kernel_is_expensive) {
            // Inline this inexpensive node.
            inline_ready->push_back(tagged_node);
        } else {
            if (curr_expensive_node) {
                // Dispatch to another thread since there is plenty of work to
                // do for this thread.
                runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node, scheduled_usec));
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
            runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node, scheduled_usec));
        }
    }
}

inline void ExecutorState::MaybeMarkCompleted(FrameState *frame, tf::int64 iter, tf::int64 node_id)
{
    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
        const NodeItem *item = impl_->gview_.node(node_id);
        tf::mutex_lock l(frame->mu);
        frame->GetIteration(iter)->mark_completed(item->pending_id);
    }
}

void ExecutorState::Finish()
{
    mu_.lock();
    auto status = status_;
    auto done_cb = std::move(done_cb_);
    auto runner = std::move(runner_);
    auto ictx = std::move(ictx_);
    mu_.unlock();
    if (sync_on_finish_ && status.ok()) {
        // Block until the device has finished all queued operations. For
        // devices like GPUs that continue to execute Ops after their Compute
        // methods have completed, this ensures that control is not returned to
        // the user until the step (and its side-effects) has actually completed.
        status = impl_->params_.device->Sync();
    }
    LogOpTracing() << "event: end_iter "
                   << nlohmann::json({{"sess", impl_->params_.session},
                                      {"graphId", impl_->graph_id_},
                                      {"stepId", step_id_},
                                      {"mainIter", impl_->is_main_iter},
                                      {"memMap", TFInstance::instance().dumpGPUMemoryMap()}});
    if (impl_->is_main_iter) {
        impl_->params_.ins->dropExlusiveMode();
        ictx->finish();
    }
    delete this;
    CHECK(done_cb != nullptr);
    runner([=]() {
        done_cb(status);
    });
}

void ExecutorState::FindOrCreateChildFrame(FrameState *frame, tf::int64 iter, const tf::Node *node,
                                           FrameState **child)
{
    // Get the child frame name.
    tf::string enter_name;
    Status s = GetNodeAttr(node->attrs(), "frame_name", &enter_name);
    DCHECK(s.ok()) << s;
    const tf::string child_name = MakeFrameName(frame, iter, enter_name);

    {
        tf::mutex_lock executor_lock(mu_);
        auto it = outstanding_frames_.find(child_name);
        if (it != outstanding_frames_.end()) {
            *child = it->second;
            return;
        }
    }

    // Need to create a new frame instance.
    // Note that this new frame instance is created without any locks.
    if (vlog_)
        VLOG(2) << "Create frame: " << child_name;

    int parallel_iters;
    s = GetNodeAttr(node->attrs(), "parallel_iterations", &parallel_iters);
    DCHECK(s.ok()) << s;
    FrameState *temp = new FrameState(impl_, parallel_iters);
    temp->frame_name = child_name;
    temp->frame_id = tf::Hash64(child_name);
    temp->parent_frame = frame;
    temp->parent_iter = iter;
    temp->InitializeFrameInfo(enter_name);

    // 'iterations' is a fixed-length circular buffer.
    temp->iterations.resize(temp->max_parallel_iterations + 1);
    // Initialize iteration 0.
    temp->iterations[0] = new IterationState(temp->pending_counts, temp->total_input_tensors);

    {
        tf::mutex_lock executor_lock(mu_);
        auto it = outstanding_frames_.find(child_name);
        if (it != outstanding_frames_.end()) {
            *child = it->second;
        } else {
            tf::mutex_lock frame_lock(frame->mu);
            frame->GetIteration(iter)->outstanding_frame_count++;
            outstanding_frames_[child_name] = temp;
            *child = temp;
            temp = nullptr;
        }
    }
    delete temp; // Not used so delete it.
}

void ExecutorState::DeleteFrame(FrameState *frame, TaggedNodeSeq *ready)
{
    // First, propagate dead_exits (if any) to the parent frame.
    FrameState *parent_frame = frame->parent_frame;
    const tf::int64 parent_iter = frame->parent_iter;
    if (parent_frame != nullptr) {
        tf::mutex_lock paranet_frame_lock(parent_frame->mu);
        // Propagate all the dead exits to the parent frame.
        for (const auto *node : frame->dead_exits) {
            auto parent_iter_state = parent_frame->GetIteration(parent_iter);
            for (const auto *e : node->out_edges()) {
                const auto *dst_node = e->dst();

                const auto dst_pending_id = impl_->gview_.node(dst_node->id())->pending_id;

                // TODO(yuanbyu): We don't need this if we require the subgraph
                // given to an executor not to contain a sink node.
                if (dst_node->IsSink())
                    continue;

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
                    if (tf::IsControlTrigger(dst_node))
                        dst_dead = false;
                    ready->push_back(TaggedNode(dst_node, parent_frame, parent_iter, dst_dead));
                    parent_iter_state->outstanding_ops++;
                }
            }
        }
    }

    // Delete the frame.
    const auto &frame_name = frame->frame_name;
    if (vlog_)
        VLOG(2) << "Delete frame " << frame_name;
    {
        tf::mutex_lock executor_lock(mu_);
        outstanding_frames_.erase(frame_name);
    }
    delete frame;
}

void ExecutorState::CleanupFramesIterations(FrameState *frame, tf::int64 iter, TaggedNodeSeq *ready)
{
    bool is_frame_done = false;
    {
        tf::mutex_lock frame_lock(frame->mu);
        frame->GetIteration(iter)->outstanding_frame_count--;
        is_frame_done = frame->CleanupIterations(&impl_->gview_, iter, ready);
    }
    if (is_frame_done) {
        FrameState *parent_frame = frame->parent_frame;
        const tf::int64 parent_iter = frame->parent_iter;
        DeleteFrame(frame, ready);
        if (parent_frame != nullptr) {
            // The completion of frame may cause completions in its parent frame.
            // So clean things up recursively.
            CleanupFramesIterations(parent_frame, parent_iter, ready);
        }
    }
}

void ExecutorState::FrameState::ActivateNodes(const NodeItem *item, const bool is_dead, tf::int64 iter,
                                              EntryVector *outputs, TaggedNodeSeq *ready)
{
    const GraphView &gview = executor->gview_;
    IterationState *iter_state = GetIteration(iter);
    const size_t num_output_edges = item->num_output_edges;
    const EdgeInfo *edges = item->output_edge_list();
    Entry *input_tensors = iter_state->input_tensors;
    for (size_t out_index = 0; out_index < num_output_edges; out_index++) {
        const EdgeInfo &e = edges[out_index];
        const int dst_id = e.dst_id;
        const NodeItem *dst_item = gview.node(dst_id);
        const auto dst_pending_id = dst_item->pending_id;
        const int src_slot = e.output_slot;

        // TODO(yuanbyu): We don't need this if we require the subgraph
        // given to an executor not to contain a sink node.
        if (dst_item->is_sink)
            continue;

        bool dst_dead = false;
        bool dst_ready = false;
        // True iff this input for dst is needed. We only set this input for
        // dst if this flag is true. This is needed to make the thread safety
        // analysis happy.
        const bool is_control_edge = (src_slot == tf::Graph::kControlSlot);
        bool dst_need_input = !is_control_edge;
        if (dst_item->is_merge) {
            // A merge node is ready if all control inputs have arrived and either
            // a) a live data input becomes available or b) all data inputs are
            // dead. For Merge, pending's LSB is set iff a live data input has
            // arrived.
            if (is_control_edge) {
                iter_state->decrement_pending(dst_pending_id, 2);
                int count = iter_state->pending(dst_pending_id);
                int dead_cnt = iter_state->dead_count(dst_pending_id);
                dst_dead = (dead_cnt == dst_item->num_inputs);
                dst_ready = (count == 0) || ((count == 1) && dst_dead);
            } else {
                if ((*outputs)[src_slot].has_value) {
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
                    dst_dead = (dead_cnt == dst_item->num_inputs) || item->is_enter;
                    dst_ready = (iter_state->pending(dst_pending_id) == 1) && dst_dead;
                    dst_need_input = false;
                }
            }
        } else {
            const bool increment_dead = (is_dead || (!is_control_edge && !(*outputs)[src_slot].has_value));
            int pending, dead;
            iter_state->adjust_for_activation(dst_pending_id, increment_dead, &pending, &dead);
            dst_dead = (dead > 0);
            dst_ready = (pending == 0);
        }

        if (dst_need_input) {
            const int dst_slot = e.input_slot;
            const int dst_loc = dst_item->input_start + dst_slot;
            if (e.is_last) {
                input_tensors[dst_loc] = std::move((*outputs)[src_slot]);
            } else {
                input_tensors[dst_loc] = (*outputs)[src_slot];
            }
        }

        // Add dst to the ready queue if it's ready
        if (dst_ready) {
            if (dst_item->is_control_trigger)
                dst_dead = false;
            ready->push_back(TaggedNode(dst_item->node, this, iter, dst_dead));
            iter_state->outstanding_ops++;
        }
    }
}

void ExecutorState::FrameState::ActivateNexts(const GraphView *gview, tf::int64 iter, TaggedNodeSeq *ready)
{
    // Propagate the deferred NextIteration nodes to the new iteration.
    for (auto &node_entry : next_iter_roots) {
        const auto *node = node_entry.first;
        const Entry &entry = node_entry.second;
        const bool is_dead = !entry.has_value;
        const NodeItem *item = gview->node(node->id());
        EntryVector outputs{entry};
        ActivateNodes(item, is_dead, iter, &outputs, ready);
    }
    next_iter_roots.clear();
}

void ExecutorState::FrameState::ActivateLoopInvs(const GraphView *gview, tf::int64 iter, TaggedNodeSeq *ready)
{
    // Propagate loop invariants to the new iteration.
    for (auto &node_entry : inv_values) {
        const auto *node = node_entry.first;
        const Entry &entry = node_entry.second;
        const bool is_dead = !entry.has_value;
        const NodeItem *item = gview->node(node->id());
        EntryVector outputs{entry};
        ActivateNodes(item, is_dead, iter, &outputs, ready);
    }
}

void ExecutorState::FrameState::AddLoopInv(const NodeItem *item, const Entry &entry, TaggedNodeSeq *ready)
{
    // Store this value.
    inv_values.push_back({item->node, entry});

    // Make this value available to all iterations.
    const bool is_dead = !entry.has_value;
    for (int i = 0; i <= iteration_count; ++i) {
        EntryVector outputs{entry};
        ActivateNodes(item, is_dead, i, &outputs, ready);
    }
}

bool ExecutorState::FrameState::IsIterationDone(tf::int64 iter)
{
    IterationState *iter_state = GetIteration(iter);
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

void ExecutorState::FrameState::IncrementIteration(const GraphView *gview, TaggedNodeSeq *ready)
{
    iteration_count++;
    const tf::int64 next_iter = iteration_count;

    // Initialize the next iteration.
    IterationState *iter_state = new IterationState(pending_counts, total_input_tensors);
    SetIteration(next_iter, iter_state);
    num_outstanding_iterations++;
    dead_exits.clear();

    // Activate the successors of the deferred roots in the new iteration.
    ActivateNexts(gview, next_iter, ready);

    // Activate the loop invariants in the new iteration.
    ActivateLoopInvs(gview, next_iter, ready);
}

bool ExecutorState::FrameState::CleanupIterations(const GraphView *gview, tf::int64 iter,
                                                  TaggedNodeSeq *ready)
{
    tf::int64 curr_iter = iter;
    while (curr_iter <= iteration_count && IsIterationDone(curr_iter)) {
        // Delete the iteration curr_iter.
        delete GetIteration(curr_iter);
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

class TFExecutorTask : public IterationTask
{
    ExecutorImpl &m_impl;
    tf::CancellationManager &m_cm;

    std::unique_ptr<ExecutorState> m_state;

public:
    TFExecutorTask(ExecutorImpl &impl, const tf::Executor::Args &args, tf::Executor::DoneCallback done)
        : m_impl(impl)
        , m_cm(*args.cancellation_manager)
        , m_state(std::make_unique<ExecutorState>(args, &impl, std::move(done)))
    {
    }

    uint64_t graphId() const override
    {
        return m_impl.graph_id_;
    }

    bool prepare() override
    {
        if (!m_impl.is_main_iter) {
            return true;
        }

        auto &ectx = m_impl.params_.ins;
        return ectx->m_item->beginIteration(ectx->m_ticket, {}, graphId());
    }

    ResStats estimatedPeakAllocation(const DeviceSpec &) const override
    {
        return {};
    }

    void runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept override
    {
        // SIExecutorState will delete itself after called runAsync
        m_state.release()->runAsync(std::move(ictx));
    }

    void cancel() override
    {
        m_cm.StartCancel();
    }

    bool isCanceled() const override
    {
        return m_cm.IsCancelled();
    }
};

void ExecutorImpl::RunAsync(const Args &args, DoneCallback done)
{
    params_.ins->scheduleIteartion(std::make_unique<TFExecutorTask>(*this, args, std::move(done)));
}

} // end namespace

Status NewTFExecutor(TFExecutorParams params, std::unique_ptr<const tf::Graph> &&graph,
                     tf::Executor **executor)
{
    auto *impl = new ExecutorImpl(params, std::move(graph));
    const Status s = impl->Initialize();
    if (s.ok()) {
        *executor = impl;
    } else {
        delete impl;
    }
    return s;
}

Status CreateNonCachedKernel(tf::Device *device, tf::FunctionLibraryRuntime *flib, const tf::NodeDef &ndef,
                             int graph_def_version, tf::OpKernel **kernel)
{
    const auto device_type = tf::DeviceType(device->attributes().device_type());
    auto allocator = device->GetAllocator(tf::AllocatorAttributes());
    return CreateOpKernel(device_type, device, allocator, flib, ndef, graph_def_version, kernel);
}

void DeleteNonCachedKernel(tf::OpKernel *kernel)
{
    delete kernel;
}

} // namespace salus::oplib::tensorflow
