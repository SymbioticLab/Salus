//
// Created by peifeng on 4/17/18.
//

#ifndef SALUS_EXEC_SCHED_SCHEDULINGPARAM_H
#define SALUS_EXEC_SCHED_SCHEDULINGPARAM_H

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
