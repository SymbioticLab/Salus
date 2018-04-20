//
// Created by peifeng on 4/16/18.
//

#ifndef SALUS_OPLIB_TENSORFLOW_COSTMGR_H
#define SALUS_OPLIB_TENSORFLOW_COSTMGR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "resources/resources.h"
#include "oplibraries/tensorflow/v2/graphview.h"
#include <boost/container/flat_map.hpp>

namespace salus::oplib::tensorflow {

/**
 * Tracking the cost of an iteration, basically an subgraph the ExecutorImpl is created for
 */
class IterationCost
{
    mutable std::mutex m_mu;
    bool isFirstIter GUARDED_BY(m_mu);
    ResStats m_iterationCost GUARDED_BY(m_mu);
    std::vector<boost::container::flat_map<DeviceType, Resources>> m_nodeCosts GUARDED_BY(m_mu);

public:
    void build(const tf::Graph &g, const GraphView &gv, const ResStats &rm);
    std::optional<Resources> getForNode(const NodeItem &item, const DeviceType &dt) const;
    ResStats getForIteration() const;

    void updateNode(const NodeItem &item, const DeviceType &dt, Resources res);
    void updateIteartion(ResStats res);
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_COSTMGR_H
