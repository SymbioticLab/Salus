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

#ifndef SALUS_EXEC_EXECUTIONENGINE_H
#define SALUS_EXEC_EXECUTIONENGINE_H

#include "devices.h"

#include "execution/resources.h"
#include "execution/threadpool/threadpool.h"
#include "platform/logging.h"
#include "utils/containerutils.h"
#include "utils/pointerutils.h"
#include "utils/threadutils.h"

#include <atomic>
#include <chrono>
#include <future>
#include <list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace salus {
class OperationTask;
} // namespace salus
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

using PSessionItem = std::shared_ptr<SessionItem>;
using POpItem = std::shared_ptr<OperationItem>;
class BaseScheduler;
class ExecutionEngine;

/**
 * @todo write docs
 */
class ExecutionContext
{
    struct Data
    {

        Data(PSessionItem &&item, uint64_t resOffer, ExecutionEngine &engine)
            : item(std::forward<PSessionItem>(item))
            , resOffer(resOffer)
            , engine(engine)
        {
        }

        ~Data();

        /**
         * @brief remove from engine and give up our reference of session item
         */
        void removeFromEngine();
        void insertIntoEngine();
        void enqueueOperation(std::unique_ptr<salus::OperationTask> &&task);

        /**
         * @brief Make a resource context that first allocate from session's resources
         * @param spec
         * @param res
         * @param missing
         * @return
         */
        std::unique_ptr<ResourceContext> makeResourceContext(const salus::DeviceSpec &spec,
                                                             const Resources &res,
                                                             Resources *missing = nullptr);

        PSessionItem item;
        uint64_t resOffer;

    private:
        ExecutionEngine &engine;
    };

    friend class ExecutionEngine;
    std::shared_ptr<Data> m_data;

public:
    ExecutionContext() = default;
    ExecutionContext(PSessionItem item, uint64_t resOffer, ExecutionEngine &engine)
        : m_data(std::make_shared<Data>(std::move(item), resOffer, engine))
    {
    }

    ExecutionContext(const ExecutionContext &) = default;
    ExecutionContext(ExecutionContext &&) = default;
    ExecutionContext &operator=(const ExecutionContext &) = default;
    ExecutionContext &operator=(ExecutionContext &&) = default;

    operator bool() const
    {
        return m_data != nullptr;
    }

    void acceptOffer(const std::string &sessHandle);

    std::optional<ResourceMap> offeredSessionResource() const;

    void enqueueOperation(std::unique_ptr<salus::OperationTask> &&task);

    void registerPagingCallbacks(PagingCallbacks &&pcb);

    void deleteSession(std::function<void()> cb);

    std::unique_ptr<ResourceContext> makeResourceContext(const salus::DeviceSpec &spec, const Resources &res,
                                                         Resources *missing = nullptr);
};

/**
 * @brief
 */
class ExecutionEngine
{

public:
    static ExecutionEngine &instance();

    ~ExecutionEngine();

    ExecutionContext createSessionOffer(ResourceMap rm);

    void startScheduler();
    void stopScheduler();

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
    ExecutionEngine() = default;

    // scheduler parameters
    SchedulingParam m_schedParam;

    std::atomic<bool> m_shouldExit{false};
    std::unique_ptr<std::thread> m_schedThread;
    void scheduleLoop();
    bool maybeWaitForAWhile(size_t scheduled);

    // Task life cycle
    friend class BaseScheduler;
    std::unique_ptr<ResourceContext> makeResourceContext(PSessionItem sess, const salus::DeviceSpec &spec,
                                                         const Resources &res, Resources *missing = nullptr);

    POpItem submitTask(POpItem &&opItem);
    void taskStopped(OperationItem &opItem, bool failed);
    void taskRunning(OperationItem &opItem);

    // Bookkeeping
    ResourceMonitor m_resMonitor;
    std::atomic_int_fast64_t m_runningTasks{0};
    std::atomic_int_fast64_t m_noPagingRunningTasks{0};

    /**
     * @brief Do paging on device 'spec'
     * @param spec
     * @param target page out to device 'target'
     * @return
     */
    bool doPaging(const salus::DeviceSpec &spec, const salus::DeviceSpec &target);

    // Incoming kernels
    void pushToSessionQueue(POpItem &&opItem);

    friend class ResourceContext;

    std::list<PSessionItem> m_newSessions;
    std::mutex m_newMu;

    std::unordered_set<PSessionItem> m_deletedSessions;
    std::mutex m_delMu;

    sstl::notification m_note_has_work;
    // Use a minimal linked list because the only operation we need is
    // iterate through the whole list, insert at end, and delete.
    // Insert and delete rarely happens, and delete is handled in the same thread
    // as iteration.
    std::list<PSessionItem> m_sessions;

    friend struct ExecutionContext::Data;
    void insertSession(PSessionItem item);
    void deleteSession(PSessionItem item);

    // Backend thread pool
    ThreadPool m_pool;
};

class ResourceContext
{
    ResourceMonitor &resMon;
    salus::DeviceSpec m_spec;

    uint64_t m_ticket;
    PSessionItem session;
    bool hasStaging;

public:
    const salus::DeviceSpec &spec() const
    {
        return m_spec;
    }
    uint64_t ticket() const
    {
        return m_ticket;
    }

    bool isGood() const
    {
        return hasStaging;
    }

    /**
     * @brief Construct a new resource context with a different spec
     * @param other
     * @param spec
     */
    ResourceContext(const ResourceContext &other, const salus::DeviceSpec &spec);

    /**
     * @brief Construct a resource context
     * @param item
     * @param resMon
     */
    ResourceContext(PSessionItem item, ResourceMonitor &resMon);
    ~ResourceContext();

    /**
     * @brief Initialize staging
     * @param spec
     * @param res
     * @param missing
     * @return
     */
    bool initializeStaging(const salus::DeviceSpec &spec, const Resources &res, Resources *missing);
    void releaseStaging();

    struct OperationScope
    {
        explicit OperationScope(const ResourceContext &context, ResourceMonitor::LockedProxy &&proxy)
            : valid(false)
            , proxy(std::move(proxy))
            , res()
            , context(context)
        {
        }

        OperationScope(OperationScope &&scope) noexcept
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

        const Resources &resources() const
        {
            return res;
        }

    private:
        void commit();

        friend class ResourceContext;

        bool valid;
        ResourceMonitor::LockedProxy proxy;
        Resources res;
        const ResourceContext &context;
    };

    /**
     * @brief Allocate all resource of type `type' in staging area
     * @param type
     * @return
     */
    OperationScope alloc(ResourceType type) const;

    OperationScope alloc(ResourceType type, size_t num) const;

    void dealloc(ResourceType type, size_t num) const;

    /**
     * @brief Called by PerOpAllocator when no allocation is hold by the ticket
     *
     */
    void removeTicketFromSession() const;
};
std::ostream &operator<<(std::ostream &os, const ResourceContext &c);

#endif // SALUS_EXEC_EXECUTIONENGINE_H
