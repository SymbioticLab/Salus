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

#include "execution/itask.h"
#include "platform/logging.h"
#include "utils/threadutils.h"
#include "utils/containerutils.h"
#include "execution/resources.h"
#include "execution/threadpool/threadpool.h"

#include <atomic>
#include <list>
#include <future>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <type_traits>

class OperationTask;
class ResourceContext;

struct SchedulingParam
{
    bool useFairnessCounter = true;
    /**
     * Maximum head-of-line waiting tasks allowed before refuse to schedule
     * later tasks
     */
    uint64_t maxHolWaiting = 50;
    /**
     * Add randomness when run tasks
     */
    bool randomizedExecution = false;
    /**
     * Whether to be work conservative. Only has effect when useFairnessCounter is true.
     */
    bool workConservative = true;
};

/**
 * @todo write docs
 */
class ExecutionEngine
{
    struct SessionItem;
    struct OperationItem;

    using PSessionItem = std::shared_ptr<SessionItem>;
    using POpItem = std::shared_ptr<OperationItem>;

    using KernelQueue = std::list<POpItem>;
    using UnsafeQueue = std::list<POpItem>;

public:
    static ExecutionEngine &instance();

    ~ExecutionEngine();

    struct PagingCallbacks
    {
        std::function<void()> forceEvicted;
        std::function<size_t(uint64_t, std::unique_ptr<ResourceContext>&&)> volunteer;

        operator bool() const {
            return forceEvicted && volunteer;
        }
    };

    class InserterImpl
    {
    public:
        InserterImpl(InserterImpl &&other)
            : InserterImpl(std::move(other.m_item), other.m_engine)
        { }
        InserterImpl(const PSessionItem &item, ExecutionEngine &engine) : m_item(item), m_engine(engine) {}

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

    ThreadPool &pool() {
        return m_pool;
    }

    void setSchedulingParam(const SchedulingParam &param) {
        m_schedParam = param;
    }

    const SchedulingParam &schedulingParam() const {
        return m_schedParam;
    }

private:
    ExecutionEngine();

    // scheduler parameters
    SchedulingParam m_schedParam;

    std::atomic<bool> m_shouldExit = {false};
    std::unique_ptr<std::thread> m_schedThread;
    void scheduleLoop();
    bool shouldWaitForAWhile(size_t scheduled, std::chrono::nanoseconds &ns);

    // Task life cycle
    size_t maybeScheduleFrom(PSessionItem item);
    bool maybePreAllocateFor(SessionItem &item, OperationItem &opItem, const DeviceSpec &spec);

    void taskStopped(SessionItem &item, OperationItem &opItem, bool failed);

    // Bookkeeping
    ResourceMonitor m_resMonitor;
    // TODO: remove this
    std::atomic_int_fast64_t m_runningTasks;

    std::atomic_int_fast64_t m_noPagingRunningTasks;
    std::unordered_map<uint64_t, std::weak_ptr<SessionItem>> m_ticketOwners;

    // Paging
    bool doPaging();

    // Incoming kernels
    struct SessionItem
    {
        // also protected by mu (may be accessed both in schedule thread and close session thread)
        PagingCallbacks pagingCb;
        std::function<void()> cleanupCb;

        std::string sessHandle;
        KernelQueue queue;
        std::mutex mu;

        // Only accessed by main scheduling thread
        UnsafeQueue bgQueue;
        double unifiedResSnapshot;
        bool forceEvicted {false};

        uint64_t holWaiting = 0;
        uint64_t queueHeadHash = 0;

        // Accessed by multiple scheduling thread
        std::atomic_bool protectOOM {true};

        std::unordered_set<uint64_t> tickets;
        std::mutex tickets_mu;

        explicit SessionItem(const std::string &handle)
            : sessHandle(handle)
        {
            // NOTE: add other devices
            resUsage[ResourceTag::GPU0Memory()].get() = 0;
            resUsage[ResourceTag::CPU0Memory()].get() = 0;
        }

        ~SessionItem();

        utils::MutableAtom::value_type &resourceUsage(const ResourceTag &tag)
        {
            return resUsage.at(tag).get();
        }

    private:
        using AtomicResUsages = std::unordered_map<ResourceTag, utils::MutableAtom>;
        // must be initialized in constructor
        AtomicResUsages resUsage;
    };
    void pushToSessionQueue(PSessionItem item, POpItem opItem);

    struct OperationItem
    {
        std::unique_ptr<OperationTask> op;

        uint64_t hash() const { return reinterpret_cast<uint64_t>(this); }

        std::chrono::time_point<std::chrono::system_clock> tQueued;
        std::chrono::time_point<std::chrono::system_clock> tInspected;
        std::chrono::time_point<std::chrono::system_clock> tScheduled;
        std::chrono::time_point<std::chrono::system_clock> tRunning;
    };
    friend class ResourceContext;

    using SessionList = std::list<PSessionItem>;
    using SessionSet = std::unordered_set<PSessionItem>;

    SessionList m_newSessions;
    std::mutex m_newMu;

    SessionSet m_deletedSessions;
    std::mutex m_delMu;

    utils::notification m_note_has_work;
    // Use a minimal linked list because the only operation we need is
    // iterate through the whole list, insert at end, and delete.
    // Insert and delete rarely happens, and delete is handled in the same thread
    // as iteration.
    SessionList m_sessions;

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
    const DeviceSpec &spec() const { return m_spec; }
    uint64_t ticket() const { return m_ticket; }

    ResourceContext(const ResourceContext &other, const DeviceSpec &spec);
    ResourceContext(ExecutionEngine::SessionItem &item, ResourceMonitor &resMon);
    ~ResourceContext();

    bool initializeStaging(const DeviceSpec &spec, const Resources &res);
    void releaseStaging();

    struct OperationScope
    {
        explicit OperationScope(const ResourceContext &context, ResourceMonitor::LockedProxy &&proxy)
            : proxy(std::move(proxy))
            , context(context)
        {}

        OperationScope(OperationScope &&scope)
            : valid(scope.valid)
            , proxy(std::move(scope.proxy))
            , context(scope.context)
        {
            scope.valid = false;
        }

        ~OperationScope() {
            commit();
        }

        operator bool() const { return valid; }

        void rollback();

    private:
        void commit();

        friend class ResourceContext;

        bool valid = true;
        ResourceMonitor::LockedProxy proxy;
        Resources res;
        const ResourceContext & context;
    };

    OperationScope allocMemory(size_t num_bytes) const;
    void deallocMemory(size_t num_bytes) const;

private:
    void removeTicketFromSession() const;

    ExecutionEngine::SessionItem &session;
    std::atomic<bool> hasStaging;
};
std::ostream &operator<<(std::ostream &os, const ResourceContext &c);

#endif // EXECUTIONENGINE_H
