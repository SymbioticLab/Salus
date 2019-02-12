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

#ifndef SALUS_EXEC_SCHED_BASESCHEDULER_H
#define SALUS_EXEC_SCHED_BASESCHEDULER_H

#include "sessionitem.h"

#include "utils/cpp17.h"
#include "utils/pointerutils.h"

#include <boost/container/small_vector.hpp>

#include <string_view>
#include <memory>
#include <mutex>
#include <map>
#include <utility>
#include <functional>

struct SessionChangeSet
{
    SessionSet deletedSessions;
    size_t numAddedSessions = 0;
    SessionList::iterator addedSessionBegin;
    SessionList::iterator addedSessionEnd;
};

namespace salus {
class TaskExecutor;
} // namespace salus

/**
 * @brief Scheduler interface used in the execution engine.
 *
 * The life time of a scheduler within the scheduling loop:
 * 1. selectCandidateSessions - beginning of a new scheduling iteration,
 *                              get notified about any session addition and removal
 * 2. maybeScheduleFrom - called for each candidate session
 */
class BaseScheduler
{
public:
    explicit BaseScheduler(salus::TaskExecutor &engine);
    virtual ~BaseScheduler();

    /**
     * @brief Name of the scheduler
     * @return name
     */
    virtual std::string name() const = 0;

    using CandidateList = boost::container::small_vector_base<PSessionItem>;
    virtual void notifyPreSchedulingIteration(const SessionList &sessions,
                                              const SessionChangeSet &changeset,
                                              sstl::not_null<CandidateList *> candidates);
    /**
     * @brief schedule from a particular session
     * @returns number of tasks scheduled, and whether should continue to next session.
     */
    virtual std::pair<size_t, bool> maybeScheduleFrom(PSessionItem item) = 0;

    /**
     * @brief Whether we should do paging in this iteration.
     *
     * Not returning true doesn't necesary triggers paging. ExecutionEngine::m_noPagingRunningTask must
     * be 0 to trigger a paging.
     *
     * @param spec Which device to consider
     *
     * @returns true if paging is needed
     */
    virtual bool insufficientMemory(const salus::DeviceSpec &spec);

    /**
     * @brief Per session debug information.
     * @param item pointer to the session
     * @returns debug information related to session `item`
     */
    virtual std::string debugString(const PSessionItem &item) const;

    /**
     * @brief debug information
     * @returns debug information
     */
    virtual std::string debugString() const;

protected:
    /**
     * @brief Preallocate resources for task on device
     *
     * Also updates internal bookkeeping of failure resources.
     *
     * @param opItem the task to preallocate
     * @param spec the device to preallocate on
     * @returns Whether the pre-allocation succeeded.
     */
    bool maybePreAllocateFor(OperationItem &opItem, const salus::DeviceSpec &spec);

    /**
     * @brief submit task for execution.
     * @param opItem the task to execute
     * @returns the task itself is submission failed, otherwise nullptr
     */
    POpItem submitTask(POpItem &&opItem);

    /**
     * @brief a convenient helper function to submit all tasks in queue from session, with HOL blocking handled.
     *
     * The queue is modified to contain any tasks left. Ordering is not changed.
     *
     * @param item The session to schedule from
     * @returns number of tasks successfully submitted
     */
    size_t submitAllTaskFromQueue(const PSessionItem &item);


    /**
     * @brief Missing resources per operation in this iteration.
     *
     */
    std::mutex m_muRes;
    std::unordered_map<sstl::not_null<OperationItem*>, Resources> m_missingRes GUARDED_BY(m_muRes);

    salus::TaskExecutor &m_taskExec;
};

inline std::ostream& operator<<(std::ostream& out, const BaseScheduler& sch)
{
    return out << sch.debugString();
}

inline std::ostream& operator<<(std::ostream& out, const std::unique_ptr<BaseScheduler>& sch)
{
    return out << sch->debugString();
}

class SchedulerRegistary final
{
public:
    SchedulerRegistary();

    ~SchedulerRegistary();

    using SchedulerFactory = std::function<std::unique_ptr<BaseScheduler>(salus::TaskExecutor&)>;
    struct Register
    {
        explicit Register(std::string_view name, SchedulerFactory factory);
    };

    std::unique_ptr<BaseScheduler> create(std::string_view name, salus::TaskExecutor &engine) const;

    static SchedulerRegistary &instance();

private:
    struct SchedulerItem
    {
        SchedulerFactory factory;

        SchedulerItem() = default;
        explicit SchedulerItem(SchedulerFactory factory) : factory(std::move(factory)) {}

    };
    mutable std::mutex m_mu;
    // NOTE: std::unordered_map doesn't support lookup using std::string_view
    std::map<std::string, SchedulerItem, std::less<>> m_schedulers;
};

#endif // SALUS_EXEC_SCHED_BASESCHEDULER_H
