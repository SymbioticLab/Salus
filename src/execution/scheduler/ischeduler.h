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

#ifndef ISCHEDULER_H
#define ISCHEDULER_H

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

/**
 * @brief Scheduler interface to schedule
 */
class ExecutionEngine;
struct SessionChangeSet
{
    SessionSet deletedSessions;
    size_t numAddedSessions = 0;
    SessionList::iterator addedSessionBegin;
    SessionList::iterator addedSessionEnd;
};

class IScheduler
{
public:
    explicit IScheduler(ExecutionEngine &engine);
    virtual ~IScheduler();

    virtual std::string name() const = 0;

    using CandidateList = boost::container::small_vector_base<PSessionItem>;
    virtual void selectCandidateSessions(const SessionList &sessions,
                                         const SessionChangeSet &changeset,
                                         utils::not_null<CandidateList*> candidates) = 0;
    /**
     * @brief schedule from a particular session
     * @returns number of tasks scheduled, and whether should continue to next session.
     */
    virtual std::pair<size_t, bool> maybeScheduleFrom(PSessionItem item) = 0;

    /**
     * @brief Per session debug information.
     * @param item pointer to the session
     * @returns debug information related to session `item`
     */
    virtual std::string debugString(const PSessionItem &item);

    /**
     * @brief debug information
     * @returns debug information
     */
    virtual std::string debugString();

protected:
    /**
     * @brief Preallocate resources for task on device
     * @param opItem the task to preallocate
     * @param spec the device to preallocate on
     */
    bool maybePreAllocateFor(OperationItem &opItem, const DeviceSpec &spec);

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

    ExecutionEngine &m_engine;
};

class SchedulerRegistary final
{
public:
    SchedulerRegistary();

    ~SchedulerRegistary();

    using SchedulerFactory = std::function<std::unique_ptr<IScheduler>(ExecutionEngine&)>;
    struct Register
    {
        explicit Register(std::string_view name, SchedulerFactory factory);
    };

    std::unique_ptr<IScheduler> create(std::string_view name, ExecutionEngine &engine) const;

    static SchedulerRegistary &instance();

private:
    struct SchedulerItem
    {
        SchedulerFactory factory;

        SchedulerItem() = default;
        explicit SchedulerItem(SchedulerFactory factory) : factory(factory) {}

    };
    mutable std::mutex m_mu;
    // NOTE: std::unordered_map doesn't support lookup using std::string_view
    std::map<std::string, SchedulerItem, std::less<>> m_schedulers;
};

#endif // ISCHEDULER_H
