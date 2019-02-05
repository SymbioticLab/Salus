/*
 * Copyright 2019 Peifeng Yu <peifeng@umich.edu>
 * 
 * This file is part of Salus
 * (see https://github.com/SymbioticLab/Salus).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
    bool m_holding = false;
    uint64_t m_currPersist = 0;
    uint64_t m_currPeak = 0;
    size_t m_count = 0;
    AllocationRegulator::Ticket m_ticket{};

    boost::circular_buffer<std::pair<long, size_t>> m_buf;

    void releaseAllocationHold();
public:
    IterAllocTracker(const ResourceTag &tag, size_t window = 0, double peakthr = 0.9);

    bool beginIter(AllocationRegulator::Ticket ticket, ResStats estimation, uint64_t currentUsage);
    bool update(size_t num);
    void endIter();
};

} // namespace salus

#endif // SALUS_MEM_ITERATIONALLOCATIONTRACKER_H
