//
// Created by peifeng on 4/16/18.
//

#include "oplibraries/tensorflow/v2/costmgr.h"
#include "utils/threadutils.h"

namespace salus::oplib::tensorflow {
void IterationCost::build(const tf::Graph &g, const GraphView &gv, const ResStats &rm, const bool is_main_iter)
{
    isFirstIter = true;
    if (is_main_iter) {
        m_iterationCost = rm;
    } else {
        m_iterationCost = {};
    }

    m_nodeCosts.resize(static_cast<size_t>(g.num_node_ids()));

    for (const auto *n : g.nodes()) {
        auto id = n->id();
        auto item = gv.node(id);
        m_nodeCosts.at(id).reserve(item->supported_devices.size());
    }
}

std::optional<Resources> IterationCost::getForNode(const NodeItem &item, const DeviceType &dt) const
{
    auto g = sstl::with_guard(m_mu);
    return sstl::optionalGet(m_nodeCosts.at(item.node->id()), dt);
}

ResStats IterationCost::getForIteration() const
{
    auto g = sstl::with_guard(m_mu);
    return m_iterationCost;
}

void IterationCost::updateNode(const NodeItem &item, const DeviceType &dt, Resources res)
{
    auto g = sstl::with_guard(m_mu);

    auto &saved = m_nodeCosts.at(item.node->id())[dt];
    if (!resources::contains(saved, res)) {
        saved = std::move(res);
    }
}

void IterationCost::updateIteartion(ResStats res)
{
    auto g = sstl::with_guard(m_mu);
    m_iterationCost = res;
    isFirstIter = false;
}
} // namespace salus::oplib::tensorflow
