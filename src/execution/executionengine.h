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
 *
 */

#ifndef EXECUTIONENGINE_H
#define EXECUTIONENGINE_H

#include "devices.h"

#include "execution/resources.h"
#include "execution/threadpool/threadpool.h"
#include "platform/logging.h"
#include "utils/containerutils.h"
#include "utils/threadutils.h"
#include "utils/pointerutils.h"

#include <atomic>
#include <chrono>
#include <future>
#include <list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

class OperationTask;
class ResourceContext;
struct SessionItem;
struct OperationItem;

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

struct PagingCallbacks
{
    std::function<void()> forceEvicted;
    std::function<size_t(uint64_t, std::unique_ptr<ResourceContext> &&)> volunteer;

    operator bool() const
    {
        return forceEvicted && volunteer;
    }
};

/**
* @todo write docs
*/
using PSessionItem = std::shared_ptr<SessionItem>;
using POpItem = std::shared_ptr<OperationItem>;
class IScheduler;
class ExecutionEngine
{

public:
    static ExecutionEngine &instance();

    ~ExecutionEngine();

    class InserterImpl
    {
    public:
        InserterImpl(InserterImpl &&other)
            : InserterImpl(std::move(other.m_item), other.m_engine)
        {
        }
        InserterImpl(const PSessionItem &item, ExecutionEngine &engine)
            : m_item(item)
            , m_engine(engine)
        {
        }

        ~InserterImpl();

        void enqueueOperation(std::unique_ptr<OperationTask> &&task);

        void registerPagingCallbacks(PagingCallbacks &&pcb);

        void deleteSession(std::function<void()> cb);

    private:
        PSessionItem m_item;
        ExecutionEngine &m_engine;
    };
    using Inserter = std::shared_ptr<InserterImpl>;

    Inserter registerSession(const std::string &sessHandle);

    ThreadPool &pool()
    {
        return m_pool;
    }

    void setSchedulingParam(const SchedulingParam &param)
    {
        m_schedParam = param;
    }

    const SchedulingParam &schedulingParam() const
    {
        return m_schedParam;
    }


private:
    ExecutionEngine();

    // scheduler parameters
    SchedulingParam m_schedParam;

    std::atomic<bool> m_shouldExit{false};
    std::unique_ptr<std::thread> m_schedThread;
    void scheduleLoop();
    bool shouldWaitForAWhile(size_t scheduled, std::chrono::nanoseconds &ns);

    // Task life cycle
    friend class IScheduler;
    bool maybePreAllocateFor(OperationItem &opItem, const DeviceSpec &spec);
    POpItem submitTask(POpItem &&opItem);
    void taskStopped(OperationItem &opItem, bool failed);
    void taskRunning(OperationItem &opItem);

    // Bookkeeping
    ResourceMonitor m_resMonitor;
    std::atomic_int_fast64_t m_runningTasks{0};
    std::atomic_int_fast64_t m_noPagingRunningTasks{0};

    // Paging
    bool doPaging();

    // Incoming kernels
    void pushToSessionQueue(POpItem &&opItem);

    friend class ResourceContext;

    std::list<PSessionItem> m_newSessions;
    std::mutex m_newMu;

    std::unordered_set<PSessionItem> m_deletedSessions;
    std::mutex m_delMu;

    utils::notification m_note_has_work;
    // Use a minimal linked list because the only operation we need is
    // iterate through the whole list, insert at end, and delete.
    // Insert and delete rarely happens, and delete is handled in the same thread
    // as iteration.
    std::list<PSessionItem> m_sessions;

    void insertSession(PSessionItem item);
    void deleteSession(PSessionItem item);

    // Backend thread pool
    ThreadPool m_pool;
};

class ResourceContext
{
    ResourceMonitor &resMon;
    DeviceSpec m_spec;
    uint64_t m_ticket;

public:
    const DeviceSpec &spec() const
    {
        return m_spec;
    }
    uint64_t ticket() const
    {
        return m_ticket;
    }

    ResourceContext(const ResourceContext &other, const DeviceSpec &spec);
    ResourceContext(SessionItem &item, ResourceMonitor &resMon);
    ~ResourceContext();

    bool initializeStaging(const DeviceSpec &spec, const Resources &res);
    void releaseStaging();

    struct OperationScope
    {
        explicit OperationScope(const ResourceContext &context, ResourceMonitor::LockedProxy &&proxy)
            : valid(true)
            , proxy(std::move(proxy))
            , res()
            , context(context)
        {
        }

        OperationScope(OperationScope &&scope)
            : valid(scope.valid)
            , proxy(std::move(scope.proxy))
            , res(std::move(scope.res))
            , context(scope.context)
        {
            scope.valid = false;
        }

        ~OperationScope()
        {
            commit();
        }

        operator bool() const
        {
            return valid;
        }

        void rollback();

    private:
        void commit();

        friend class ResourceContext;

        bool valid;
        ResourceMonitor::LockedProxy proxy;
        Resources res;
        const ResourceContext &context;
    };

    OperationScope allocMemory(size_t num_bytes) const;
    void deallocMemory(size_t num_bytes) const;

private:
    void removeTicketFromSession() const;

    SessionItem &session;
    std::atomic<bool> hasStaging;
};
std::ostream &operator<<(std::ostream &os, const ResourceContext &c);

#endif // EXECUTIONENGINE_H
