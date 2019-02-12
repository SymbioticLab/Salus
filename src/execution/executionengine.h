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

#ifndef SALUS_EXEC_EXECUTIONENGINE_H
#define SALUS_EXEC_EXECUTIONENGINE_H

#include "execution/devices.h"
#include "execution/engine/taskexecutor.h"
#include "execution/scheduler/schedulingparam.h"
#include "execution/threadpool/threadpool.h"
#include "platform/logging.h"
#include "resources/resources.h"
#include "utils/containerutils.h"
#include "utils/pointerutils.h"
#include "utils/threadutils.h"

#include <concurrentqueue.h>

#include <boost/circular_buffer.hpp>

#include <atomic>
#include <any>
#include <chrono>
#include <future>
#include <list>
#include <memory>
#include <unordered_map>
#include <set>

namespace salus {
class IterationTask;
class ExecutionContext;

class ExecutionEngine
{

public:
    static ExecutionEngine &instance();

    ~ExecutionEngine();

    void startScheduler();
    void stopScheduler();

    ThreadPool &pool()
    {
        return m_pool;
    }

    void setSchedulingParam(const salus::SchedulingParam &param)
    {
        m_schedParam = param;
    }

    const salus::SchedulingParam &schedulingParam() const
    {
        return m_schedParam;
    }

    std::shared_ptr<ExecutionContext> makeContext();

private:
    friend class ExecutionContext;

    ExecutionEngine();

    // scheduler parameters
    salus::SchedulingParam m_schedParam;

    ThreadPool m_pool;

    ResourceMonitor m_resMonitor;
    AllocationRegulator m_allocReg;

    // Task executor
    salus::TaskExecutor m_taskExecutor;

    // Iteration scheduling
    std::mutex m_mu;

    struct IterationItem
    {
        std::weak_ptr<ExecutionContext> wectx;
        std::unique_ptr<IterationTask> iter;
    };


    using IterQueue = std::list<IterationItem>;
    using BlockingQueues =
        boost::circular_buffer<std::pair<PSessionItem, boost::circular_buffer<IterationItem>>>;

    struct LaneQueue
    {
        uint64_t id;
        IterQueue queue;
        std::chrono::system_clock::time_point lastSeen;
        std::atomic_int_fast64_t numExpensiveIterRunning {0};
        std::set<std::weak_ptr<SessionItem>, std::owner_less<std::weak_ptr<SessionItem>>> sessions;
        SessionItem *lastSessionItem = nullptr;
        std::list<std::weak_ptr<SessionItem>> fifoQueue;
    };

    IterQueue m_iterQueue GUARDED_BY(m_mu);
    void scheduleIteration(IterationItem &&item);

    std::unique_ptr<std::thread> m_schedThread;
    std::atomic<bool> m_interrupting{false};
    sstl::notification m_note_has_work;

    void scheduleLoop();
    int scheduleOnQueue(LaneQueue &lctx, IterQueue &staging);
    bool checkIter(IterationItem &iterItem, ExecutionContext &ectx, LaneQueue &lctx);
    bool runIter(IterationItem &iterItem, ExecutionContext &ectx, LaneQueue &lctx);
    bool maybeWaitForAWhile(size_t scheduled);
    void maybeWaitForWork(size_t pending, size_t scheduled);
};

/**
 * @todo write docs
 */
class ExecutionContext : public std::enable_shared_from_this<ExecutionContext>
{
    ExecutionEngine &m_engine;
    std::any m_userData;
    uint64_t m_laneId;

    friend class ExecutionEngine;
    /**
     * @brief remove from engine and give up our reference of session item
     */
    void removeFromEngine();

public:
    AllocationRegulator::Ticket m_ticket;
    PSessionItem m_item;

    explicit ExecutionContext(ExecutionEngine &engine, AllocationRegulator::Ticket ticket);

    ~ExecutionContext()
    {
        removeFromEngine();
    }

    void scheduleIteartion(std::unique_ptr<IterationTask> &&iterTask);

    void registerPagingCallbacks(PagingCallbacks &&pcb);
    void setInterruptCallback(std::function<void()> cb);

    void setSessionHandle(const std::string &h);

    const std::any &userData() const
    {
        return m_userData;
    }

    void setUserData(std::any &&data)
    {
        m_userData = std::move(data);
    }

    void dropExlusiveMode();

    uint64_t laneId() const
    {
        return m_laneId;
    }

    void setLaneId(uint64_t id)
    {
        m_laneId = id;
    }

    void setExpectedRunningTime(uint64_t time);

    /**
     * @brief Make a resource context that first allocate from session's resources
     * @param spec
     * @param res
     * @param missing
     * @return
     */
    std::unique_ptr<ResourceContext> makeResourceContext(uint64_t graphId, const DeviceSpec &spec,
                                                         const Resources &res, Resources *missing = nullptr);

    void finish(std::function<void()> cb);
};

} // namespace salus

#endif // SALUS_EXEC_EXECUTIONENGINE_H
