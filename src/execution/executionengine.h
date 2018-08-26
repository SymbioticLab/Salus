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

#include "execution/devices.h"
#include "execution/engine/taskexecutor.h"
#include "execution/scheduler/schedulingparam.h"
#include "execution/threadpool/threadpool.h"
#include "platform/logging.h"
#include "resources/resources.h"
#include "utils/containerutils.h"
#include "utils/pointerutils.h"
#include "utils/threadutils.h"
#include "utils/fixed_function.hpp"

#include <concurrentqueue.h>

#include <boost/circular_buffer.hpp>

#include <atomic>
#include <any>
#include <chrono>
#include <future>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>

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

    using RequestContextCb = sstl::FixedFunction<void(std::shared_ptr<ExecutionContext>)>;
    void requestContext(RequestContextCb &&done);

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
    IterQueue m_iterQueue GUARDED_BY(m_mu);
    std::atomic_int_fast64_t m_numExpensiveIterRunning {0};
    void scheduleIteration(IterationItem &&item);

    std::unique_ptr<std::thread> m_schedThread;
    std::atomic<bool> m_interrupting{false};
    sstl::notification m_note_has_work;

    void scheduleLoop();
    bool checkIter(IterationItem &iterItem, ExecutionContext &ectx);
    bool runIter(IterationItem &iterItem, ExecutionContext &ectx);
    bool maybeWaitForAWhile(size_t scheduled);
    void maybeWaitForWork(const BlockingQueues &blockingSessions, const IterQueue &iters, size_t scheduled);
};

/**
 * @todo write docs
 */
class ExecutionContext : public std::enable_shared_from_this<ExecutionContext>
{
    ExecutionEngine &m_engine;
    std::any m_userData;

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
