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
#include "execution/resources.h"

#include <q/lib.hpp>
#include <q/promise.hpp>
#include <q/execution_context.hpp>
#include <q/threadpool.hpp>

#include <list>
#include <memory>
#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include <future>
#include <chrono>

class OperationTask;
struct ResourceContext;

/**
 * @todo write docs
 */
class ExecutionEngine
{
    struct SessionItem;
    struct OperationItem;

    using KernelQueue = std::list<std::shared_ptr<OperationItem>>;
    using UnsafeQueue = std::list<std::shared_ptr<OperationItem>>;

public:
    static ExecutionEngine &instance();

    ~ExecutionEngine();

    struct PagingCallbacks
    {
        std::function<void(uint64_t, void*)> forceEvicted;
        std::function<bool(uint64_t, std::shared_ptr<ResourceContext>&&)> volunteer;

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
        InserterImpl(const std::shared_ptr<SessionItem> &item, ExecutionEngine &engine) : m_item(item), m_engine(engine) {}

        ~InserterImpl();

        std::future<void> enqueueOperation(std::unique_ptr<OperationTask> &&task);

        void registerPagingCallbacks(PagingCallbacks &&pcb);

    private:
        std::shared_ptr<SessionItem> m_item;
        ExecutionEngine &m_engine;
    };
    using Inserter = std::shared_ptr<InserterImpl>;

    Inserter registerSession(const std::string &sessHandle);

    template<typename ResponseType>
    q::promise<std::unique_ptr<ResponseType>> enqueue(PTask &&task)
    {
        using PResponse = std::unique_ptr<ResponseType>;

        return q::make_promise_of<PResponse>(m_qec->queue(),
                                             [this, task = std::move(task)](auto resolve,
                                                                            auto reject){
            try {
                if (this->schedule(task.get())) {
                    if (task->isAsync()) {
                        task->runAsync<ResponseType>([resolve](PResponse &&ptr){
                            resolve(std::move(ptr));
                        });
                    } else {
                        resolve(task->run<ResponseType>());
                    }
                } else {
                    reject(std::logic_error("Task failed to prepare"));
                }
            } catch (std::exception &err) {
                reject(err);
            }
        });
    }

    template<typename ResponseType>
    q::promise<std::unique_ptr<ResponseType>> emptyPromise()
    {
        using PResponse = std::unique_ptr<ResponseType>;
        return q::with(m_qec->queue(), PResponse(nullptr));
    }

    template<typename ResponseType>
    q::promise<ResponseType> makePromise(ResponseType &&t)
    {
        return q::with(m_qec->queue(), std::move(t));
    }

    template<typename ResponseType>
    q::promise<ResponseType> makePromise(const ResponseType &t)
    {
        return q::with(m_qec->queue(), t);
    }

protected:
    bool schedule(ITask *t);

    bool trySchedule(ITask *t, const DeviceSpec &dev);

private:
    ExecutionEngine();

    std::atomic<bool> m_shouldExit = {false};
    std::unique_ptr<std::thread> m_schedThread;
    void scheduleLoop();
    bool shouldWaitForAWhile(size_t scheduled, std::chrono::nanoseconds &ns);

    // Task life cycle
    size_t maybeScheduleFrom(std::shared_ptr<SessionItem> item);
    bool maybePreAllocateFor(SessionItem &item, OperationItem &opItem, const DeviceSpec &spec);

    void taskStopped(SessionItem &item, OperationItem &opItem);

    // Bookkeeping
    ResourceMonitor m_resMonitor;
    std::atomic_int_fast64_t m_runningTasks;
    std::unordered_map<uint64_t, std::weak_ptr<SessionItem>> m_ticketOwners;

    // Paging
    void doPaging();

    // Incoming kernels
    // Use a minimal linked list because the only operation we need is
    // iterate through the whole list, insert at end, and delete.
    // Insert and delete rarely happens, and delete is handled in the same thread
    // as iteration.
    struct SessionItem
    {
        std::string sessHandle;
        KernelQueue queue;
        std::mutex mu;

        PagingCallbacks pagingCb;

        // Only accessed by main scheduling thread
        UnsafeQueue bgQueue;

        // Accessed by multiple scheduling thread
        std::unordered_set<uint64_t> tickets;
        std::mutex tickets_mu;

        explicit SessionItem(const std::string &handle) : sessHandle(handle) {}

        ~SessionItem();

    };
    void pushToSessionQueue(std::shared_ptr<SessionItem> item, std::shared_ptr<OperationItem> opItem);

    struct OperationItem
    {
        std::unique_ptr<OperationTask> op;

        std::promise<void> promise;
        std::shared_ptr<ResourceContext> rctx;
    };
    friend struct ResourceContext;

    using SessionList = std::list<std::shared_ptr<SessionItem>>;
    using SessionSet = std::unordered_set<std::shared_ptr<SessionItem>>;

    SessionList m_newSessions;
    std::mutex m_newMu;

    SessionSet m_deletedSessions;
    std::mutex m_delMu;

    utils::notification m_note_has_work;
    SessionList m_sessions;

    void insertSession(std::shared_ptr<SessionItem> item);
    void deleteSession(std::shared_ptr<SessionItem> item);

    // Backend thread pool
    using qExecutionContext = q::specific_execution_context_ptr<q::threadpool>;
    q::scope m_qscope;
    qExecutionContext m_qec;
};

struct ResourceContext
{
    ResourceMonitor &resMon;
    DeviceSpec spec;
    uint64_t ticket;

    ResourceContext(ExecutionEngine::SessionItem &item, ResourceMonitor &resMon);
    ~ResourceContext();

    bool initializeStaging(const DeviceSpec &spec, const Resources &res);
    void releaseStaging();

    bool allocMemory(size_t num_bytes);
    void deallocMemory(size_t num_bytes);

private:
    ExecutionEngine::SessionItem &session;
};


#endif // EXECUTIONENGINE_H
