/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017  Aetf <aetf@unlimitedcodeworks.xyz>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

class ExecutionEngine;
struct SessionChangeSet
{
    SessionSet deletedSessions;
    size_t numAddedSessions = 0;
    SessionList::iterator addedSessionBegin;
    SessionList::iterator addedSessionEnd;
};

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
    explicit BaseScheduler(ExecutionEngine &engine);
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

    ExecutionEngine &m_engine;
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

    using SchedulerFactory = std::function<std::unique_ptr<BaseScheduler>(ExecutionEngine&)>;
    struct Register
    {
        explicit Register(std::string_view name, SchedulerFactory factory);
    };

    std::unique_ptr<BaseScheduler> create(std::string_view name, ExecutionEngine &engine) const;

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
