//
// Created by peifeng on 4/20/18.
//

#ifndef SALUS_MEM_ITERATIONALLOCATIONTRACKER_H
#define SALUS_MEM_ITERATIONALLOCATIONTRACKER_H

#include "resources/resources.h"

#include <boost/circular_buffer.hpp>

namespace salus {

class IterAllocTracker
{
    // knobs
    ResourceTag m_tag;
    double m_peakthr;
    size_t m_window;

    // cross iter state
    int m_numIters = 0;
    ResStats m_est{};
    // in iter state
    bool m_inIter = 0;
    ResStats m_curr{};
    AllocationRegulator::Ticket m_ticket{};

    boost::circular_buffer<std::pair<long, size_t>> m_buf;

public:
    IterAllocTracker(const ResourceTag &tag, size_t window = 0, double peakthr = 0.9);

    bool beginIter(AllocationRegulator::Ticket ticket, ResStats estimation);
    bool update(size_t num);
    void endIter();
};

} // namespace salus

#endif // SALUS_MEM_ITERATIONALLOCATIONTRACKER_H
