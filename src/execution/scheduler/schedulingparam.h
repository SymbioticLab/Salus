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

#ifndef SALUS_EXEC_SCHED_SCHEDULINGPARAM_H
#define SALUS_EXEC_SCHED_SCHEDULINGPARAM_H

#include <cstdint>
#include <string>

namespace salus {
struct SchedulingParam
{
    /**
     * Maximum head-of-line waiting tasks allowed before refuse to schedule
     * later tasks in the same queue.
     */
    uint64_t maxHolWaiting = 50;
    /**
     * Whether to be work conservative. This has no effect when using scheduler 'pack'
     */
    bool workConservative = true;
    /**
     * The scheduler to use
     */
    std::string scheduler = "fair";
};

} // namespace salus

#endif // SALUS_EXEC_SCHED_SCHEDULINGPARAM_H
