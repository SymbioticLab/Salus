//
// Created by peifeng on 4/17/18.
//

#ifndef SALUS_EXEC_TASKEXECUTOR_H
#define SALUS_EXEC_TASKEXECUTOR_H

#include "execution/scheduler/schedulingparam.h"
#include "resources/resources.h"
#include "utils/threadutils.h"

#include <atomic>
#include <thread>
#include <list>
#include <memory>

class ResourceMonitor;
class ThreadPool;
struct SessionItem;
using PSessionItem = std::shared_ptr<SessionItem>;
struct OperationItem;
using POpItem = std::shared_ptr<OperationItem>;
namespace salus {

class ResourceContext;
class IterationContext;
struct PagingCallbacks
{
    std::function<size_t(uint64_t, std::unique_ptr<ResourceContext> &&)> volunteer;

    operator bool() const // NOLINT
    {
        return volunteer != nullptr;
    }
};

class TaskExecutor
{
public:
    explicit TaskExecutor(ThreadPool &pool, ResourceMonitor &resMonitor, SchedulingParam &param);

    void startExecution();
    void stopExecution();

    const SchedulingParam &schedulingParam() const
    {
        return m_schedParam;
    }

    PSessionItem insertSession();

    /**
     * @brief Make a resource context that first allocate from session's resources
     * @param spec
     * @param res
     * @param missing
     * @return
     */
    std::unique_ptr<ResourceContext> makeResourceContext(PSessionItem sess,
                                                         const std::string &graphId,
                                                         const DeviceSpec &spec,
                                                         const Resources &res, Resources *missing = nullptr);

    // Incoming kernels
    void queueTask(POpItem &&opItem);

    // actually run task
    POpItem runTask(POpItem &&opItem);

    void deleteSession(PSessionItem item);

private:
    friend class BaseScheduler;

    ResourceMonitor &m_resMonitor;
    ThreadPool &m_pool;
    SchedulingParam &m_schedParam;

    // Scheduling thread control
    std::atomic<bool> m_interrupting{false};
    std::atomic<bool> m_shouldExit{false};
    std::unique_ptr<std::thread> m_schedThread;
    sstl::notification m_note_has_work;

    void scheduleLoop();
    bool maybeWaitForAWhile(size_t scheduled);

    // Sessions
    std::list<PSessionItem> m_newSessions GUARDED_BY(m_newMu);
    std::mutex m_newMu;
    std::unordered_set<PSessionItem> m_deletedSessions GUARDED_BY(m_delMu);
    std::mutex m_delMu;
    /**
     * @brief Use a minimal linked list because the only operation we need is
     * iterate through the whole list, insert at end, and delete.
     * Insert and delete rarely happens, and delete is handled in the same thread
     * as iteration.
     */
    std::list<PSessionItem> m_sessions;

    // Task life cycle
    void taskStopped(OperationItem &opItem, bool failed);
    void taskRunning(OperationItem &opItem);

    std::atomic_int_fast64_t m_nRunningTasks{0};
    std::atomic_int_fast64_t m_nNoPagingRunningTasks{0};

    /**
     * @brief Do paging on device 'spec'
     * @param spec
     * @param target page out to device 'target'
     * @return
     */
    bool doPaging(const DeviceSpec &spec, const DeviceSpec &target);
};

} // namespace salus

#endif // SALUS_EXEC_TASKEXECUTOR_H
