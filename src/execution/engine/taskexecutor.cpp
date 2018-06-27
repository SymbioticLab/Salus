//
// Created by peifeng on 4/17/18.
//

#include "execution/engine/taskexecutor.h"

#include "execution/engine/resourcecontext.h"
#include "execution/engine/iterationcontext.h"
#include "execution/operationtask.h"
#include "resources/resources.h"
#include "execution/threadpool/threadpool.h"
#include "execution/scheduler/basescheduler.h"
#include "execution/scheduler/operationitem.h"
#include "utils/date.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
using FpSeconds = std::chrono::duration<double, seconds::period>;
using namespace std::chrono_literals;
using namespace date;

namespace salus {

namespace {
inline void logScheduleFailure(const Resources &usage, const ResourceMonitor &resMon)
{
    UNUSED(usage);
    UNUSED(resMon);

#ifndef NDEBUG
    VLOG(2) << "Try to allocate resource failed. Requested: " << usage;
    // Don't call resMon.DebugString directly in log line, as logging acquires lock, and
    // may causing deadlock.
    const auto &str = resMon.DebugString();
    VLOG(2) << "Available: " << str;
#endif
}

void reportNoProgress(bool noProgress)
{
    static auto lastProgress = system_clock::now();
    if (noProgress) {
        auto dur = system_clock::now() - lastProgress;
        if (dur > 10s) {
            LOG(ERROR) << "No progress for " << dur << " check the program!!!";
        }
    } else {
        lastProgress = system_clock::now();
    }
}

} // namespace

TaskExecutor::TaskExecutor(ThreadPool &pool, ResourceMonitor &resMonitor, SchedulingParam &param)
    : m_resMonitor(resMonitor)
    , m_pool(pool)
    , m_schedParam(param)
{
}

void TaskExecutor::startExecution()
{
    // Start scheduling thread
    m_schedThread = std::make_unique<std::thread>(std::bind(&TaskExecutor::scheduleLoop, this));
}

void TaskExecutor::stopExecution()
{
    m_interrupting = true;

    // unblock scheduling thread
    m_note_has_work.notify();

    if (m_schedThread->joinable()) {
        m_schedThread->join();
    }
}

void TaskExecutor::insertSession(PSessionItem sess)
{
    if (m_interrupting) {
        return;
    }

    {
        auto g = sstl::with_guard(m_newMu);
        m_newSessions.emplace_back(sess);
    }
    m_note_has_work.notify();
}

void TaskExecutor::deleteSession(PSessionItem item)
{
    {
        auto g = sstl::with_guard(m_delMu);
        m_deletedSessions.emplace(std::move(item));
    }
    m_note_has_work.notify();
}

void TaskExecutor::queueTask(POpItem &&opItem)
{
    auto item = opItem->sess.lock();
    if (!item) {
        // session already deleted, discard this task sliently
        return;
    }

    item->queueTask(std::move(opItem));
    m_note_has_work.notify();
}

void TaskExecutor::scheduleLoop()
{
    auto scheduler = SchedulerRegistary::instance().create(m_schedParam.scheduler, *this);
    DCHECK(scheduler);
    VLOG(2) << "Using scheduler: " << scheduler;
    LOG(INFO) << "TaskExecutor scheduling thread started";

    m_nRunningTasks = 0;
    m_nNoPagingRunningTasks = 0;

    size_t schedIterCount = 0;
    boost::container::small_vector<PSessionItem, 5> candidates;
    bool interrupted = false;

    while (!m_shouldExit) {
        SessionChangeSet changeset;
        // Fisrt check if there's any pending deletions
        {
            auto g = sstl::with_guard(m_delMu);

            using std::swap;
            swap(changeset.deletedSessions, m_deletedSessions);
            DCHECK(m_deletedSessions.empty());
        }

        // Delete sessions as requested
        // NOTE: don't clear del yet, we need that in changeset for scheduling
        m_sessions.remove_if([&changeset](auto sess) {
            bool deleted = changeset.deletedSessions.count(sess) > 0;
            if (deleted) {
                VLOG(2) << "Deleting session " << sess->sessHandle << "@" << as_hex(sess);
                sess->cleanupCb();
                // reset cb to release anything that may depend on this
                // before going out of destructor.
                sess->cleanupCb = nullptr;

                // The deletion of session's executor is async to this thread.
                // So it's legit for tickets to be nonempty
                // DCHECK(item->tickets.empty());
            }
            return deleted;
        });

        // Append any new sessions
        {
            auto g = sstl::with_guard(m_newMu);

            changeset.numAddedSessions = m_newSessions.size();

            // list::splice doesn't invalidate iterators, so use
            // m_newSessions.begin() here is ok, and a must.
            changeset.addedSessionBegin = m_newSessions.begin();
            changeset.addedSessionEnd = m_sessions.end();

            m_sessions.splice(m_sessions.end(), m_newSessions);
            DCHECK(m_newSessions.empty());
        }

        if (m_interrupting && !interrupted) {
            interrupted = true;
            // Request interrupt on any existing sessions
            for (const auto &sess : m_sessions) {
                sess->interrupt();
            }
        }

        // Prepare session ready for this iter of schedule:
        // - move from front end queue to backing storage
        // - reset lastScheduled
        size_t totalRemainingCount = 0;

        // since iteration based execution, we can enable this
        const bool enableOOMProtect = true;
        for (auto &item : m_sessions) {
            {
                auto g = sstl::with_guard(item->mu);
                item->bgQueue.splice(item->bgQueue.end(), item->queue);
            }

            if (item->forceEvicted) {
                VLOG(2) << "Canceling pending tasks in forced evicted seesion: " << item->sessHandle;
                // cancel all pending tasks
                for (auto &opItem : item->bgQueue) {
                    opItem->op->cancel();
                }
                item->bgQueue.clear();
            }

            totalRemainingCount += item->bgQueue.size();

            item->protectOOM = enableOOMProtect;
            item->lastScheduled = 0;
        }

        if (interrupted) {
            // only do session acception and deletion if interrupted
            changeset.deletedSessions.clear();
            if (m_sessions.empty()) {
                break;
            } else {
                LOG(INFO) << "Waiting for " << m_sessions.size() << " sessions to finish";
            }
            continue;
        }

        // Select and sort candidates.
        scheduler->notifyPreSchedulingIteration(m_sessions, changeset, &candidates);

        // Deleted sessions are no longer needed, release them.
        changeset.deletedSessions.clear();

        // Schedule tasks from candidate sessions
        // NOTE: remainingCount only counts for candidate sessions in this sched iter.
        size_t remainingCount = 0;
        size_t scheduled = 0;
        for (auto &item : candidates) {
            VLOG(3) << "Scheduling all opItem in session " << item->sessHandle << ": queue size "
                    << item->bgQueue.size();

            // Try schedule from this session
            auto [count, shouldContinue] = scheduler->maybeScheduleFrom(item);
            item->lastScheduled = count;

            remainingCount += item->bgQueue.size();
            scheduled += item->lastScheduled;

            if (!shouldContinue) {
                break;
            }
        }

        // Log performance counters
        CLOG(INFO, logging::kPerfTag)
            << "Scheduler iter stat: " << schedIterCount << " running: " << m_nRunningTasks
            << " noPageRunning: " << m_nNoPagingRunningTasks;
        for (auto &item : m_sessions) {
            CLOG(INFO, logging::kPerfTag)
                << "Sched iter " << schedIterCount << " session: " << item->sessHandle
                << " pending: " << item->bgQueue.size() << " scheduled: " << item->lastScheduled << " "
                << scheduler->debugString(item);
        }

        // Update conditions and check if we need paging
        bool noProgress = remainingCount > 0 && scheduled == 0 && m_nNoPagingRunningTasks == 0;
        reportNoProgress(noProgress);

        noProgress = false;
        bool didPaging = false;
        // TODO: we currently assume we are paging GPU memory to CPU
        for (const auto &dev : {devices::GPU0}) {
            if (noProgress && scheduler->insufficientMemory(dev)) {
                if (m_sessions.size() > 1) {
                    didPaging = doPaging(dev, devices::CPU0);
                } else if (m_sessions.size() == 1) {
                    LOG(ERROR) << "OOM on device " << dev
                               << " for single session happened: " << m_sessions.front()->sessHandle;
                    {
                        auto g = sstl::with_guard(m_sessions.front()->tickets_mu);
                        auto usage = m_resMonitor.queryUsages(m_sessions.front()->tickets);
                        LOG(ERROR) << "This session usage:" << resources::DebugString(usage);
                    }
                    LOG(ERROR) << m_resMonitor.DebugString();
                }
            }
        }
        // succeed, retry another sched iter immediately
        if (didPaging) {
            continue;
        }

        maybeWaitForAWhile(scheduled);

        if (!totalRemainingCount) {
            VLOG(2) << "TaskExecutor wait on m_note_has_work";
            m_note_has_work.wait();
        }
    }

    // Cleanup
    CHECK(m_deletedSessions.empty());
    CHECK(m_newSessions.empty());
    CHECK(m_sessions.empty());
    LOG(INFO) << "TaskExecutor stopped";
}

bool TaskExecutor::maybeWaitForAWhile(size_t scheduled)
{
    static constexpr auto initialSleep = 10ms;
    static constexpr auto getBored = 20ms;

    static auto last = system_clock::now();
    static auto sleep = initialSleep;

    auto now = system_clock::now();

    if (scheduled > 0) {
        last = now;
        sleep = initialSleep;
    }

    auto idle = now - last;
    if (idle <= getBored) {
        return false;
    }

    VLOG(2) << "No progress for " << duration_cast<milliseconds>(idle).count() << "ms, sleep for "
            << duration_cast<milliseconds>(sleep).count() << "ms";

    // no progress for a long time.
    // give out our time slice to avoid using too much cycles
    //             std::this_thread::yield();
    std::this_thread::sleep_for(sleep);

    // Next time we'll sleep longer
    sleep *= 2;

    return true;
}

POpItem TaskExecutor::runTask(POpItem &&opItem)
{
    auto item = opItem->sess.lock();
    if (!item) {
        // discard
        return nullptr;
    }

    // NOTE: this is waited by schedule thread, so we can't afford running
    // the operation inline. If the thread pool is full, simply consider the
    // opItem as not scheduled.

    // opItem has to be captured by value, we need it in case the thread pool is full
    auto c = m_pool.tryRun([opItem, this]() mutable {
        DCHECK(opItem);

        if (auto item = opItem->sess.lock()) {
            OperationTask::Callbacks cbs;

            // capture an session item untile done
            cbs.done = [item, opItem, this]() {
                // succeed
                taskStopped(*opItem, false);
            };
            cbs.memFailure = [opItem, this]() mutable {
                auto item = opItem->sess.lock();
                if (!item) {
                    VLOG(2) << "Found expired session during handling of memory failure of opItem: "
                            << opItem->op;
                    return false;
                }
                if (opItem->op->hasExactEstimation(opItem->op->resourceContext().spec())
                    && !item->protectOOM) {
                    VLOG(2) << "Pass through OOM failed task back to client: " << opItem->op;
                    return false;
                }

                taskStopped(*opItem, true);
                // failed due to OOM. Push back to queue and retry later
                VLOG(2) << "Putting back OOM failed task: " << opItem->op;
                queueTask(std::move(opItem));
                return true;
            };

            VLOG(2) << "Running opItem in session " << item->sessHandle << ": " << opItem->op;
            taskRunning(*opItem);
            opItem->op->run(std::move(cbs));
        }
    });
    if (!c) {
        // successfully sent to thread pool, we can reset opItem
        opItem.reset();
    }
    return opItem;
}

void TaskExecutor::taskRunning(OperationItem &opItem)
{
    LogOpTracing() << "OpItem Event " << opItem.op << " event: running";
    m_nRunningTasks += 1;
    if (!opItem.op->isAsync()) {
        m_nNoPagingRunningTasks += 1;
    }
}

void TaskExecutor::taskStopped(OperationItem &opItem, bool failed)
{
    auto &rctx = opItem.op->resourceContext();
    rctx.releaseStaging();

    LogOpTracing() << "OpItem Event " << opItem.op << "failed=" << opItem.op->failedTimes() << " event: done";
    LOG(INFO) << "OpItem Event " << opItem.op << "failed=" << opItem.op->failedTimes() << " event: done";
    if (!failed) {
        if (VLOG_IS_ON(2)) {
            if (auto item = opItem.sess.lock()) {
                auto g = sstl::with_guard(item->mu);
                ++item->totalExecutedOp;
            }
        }
    }

    m_nRunningTasks -= 1;
    if (!opItem.op->isAsync()) {
        m_nNoPagingRunningTasks -= 1;
    }
}

bool TaskExecutor::doPaging(const DeviceSpec &spec, const DeviceSpec &target)
{
    auto now = system_clock::now();
    size_t released = 0;
    std::string forceEvicitedSess;

    sstl::ScopeGuards sg([&now, &released, &forceEvicitedSess]() {
        auto dur = system_clock::now() - now;
        CLOG(INFO, logging::kPerfTag)
            << "Paging: "
            << " duration: " << duration_cast<microseconds>(dur).count() << " us"
            << " released: " << released << " forceevict: '" << forceEvicitedSess << "'";
    });

    const ResourceTag srcTag{ResourceType::MEMORY, spec};
    const ResourceTag dstTag{ResourceType::MEMORY, target};

    // Step 1: select candidate sessions
    std::vector<std::pair<size_t, std::reference_wrapper<PSessionItem>>> candidates;
    candidates.reserve(m_sessions.size());

    // Step 1.1: count total memory usage for each session
    for (auto &pSess : m_sessions) {
        candidates.emplace_back(pSess->resourceUsage(srcTag), pSess);
    }

    // sort in decending order
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });

    // Step 1.2: keep the session with largest memory usage, and try from next
    // no need to erase the first elem, as it's a O(n) operation on vector

    if (candidates.size() <= 1) {
        LOG(ERROR) << "Out of memory for one session";
        return false;
    }

    if (VLOG_IS_ON(2)) {
        for (auto [usage, pSess] : candidates) {
            VLOG(2) << "Session " << pSess.get()->sessHandle << " usage: " << usage;
        }
    }

    // Step 2: inform owner to do paging given suggestion
    for (size_t i = 1; i != candidates.size(); ++i) {
        auto &pSess = candidates[i].second.get();
        std::vector<std::pair<size_t, uint64_t>> victims;
        {
            auto g = sstl::with_guard(pSess->tickets_mu);
            if (pSess->tickets.empty()) {
                // no need to go beyond
                break;
            }
            victims = m_resMonitor.sortVictim(pSess->tickets);
        }

        // we will be doing paging on this session. Lock it's input queue lock
        // also prevents the executor from clearing the paging callbacks.
        // This should not create deadlock as nothing could finish at this time,
        // thus no new tasks could be submitted.
        auto g = sstl::with_guard(pSess->mu);
        if (!pSess->pagingCb) {
            continue;
        }

        VLOG(2) << "Visiting session: " << pSess->sessHandle;

        for (auto [usage, victim] : victims) {
            // preallocate some CPU memory for use.
            Resources res{{dstTag, usage}};

            auto rctx = makeResourceContext(pSess, 0, target, res);
            if (!rctx) {
                LOG(ERROR) << "No enough CPU memory for paging. Required: " << res[dstTag] << " bytes";
                return false;
            }
            LogAlloc() << "Pre allocated " << *rctx << " for session=" << pSess->sessHandle;

            VLOG(2) << "    request to page out ticket " << victim << " of usage " << usage;
            // request the session to do paging
            released += pSess->pagingCb.volunteer(victim, std::move(rctx));
            if (released > 0) {
                // someone freed some memory on GPU, we are good to go.
                VLOG(2) << "    released " << released << " bytes via paging";
                return true;
            }
            VLOG(2) << "    failed";
        }
        // continue to next session
    }

    LOG(ERROR) << "All paging request failed. Dump all session usage";
    for (auto [usage, pSess] : candidates) {
        LOG(ERROR) << "Session " << pSess.get()->sessHandle << " usage: " << usage;
    }
    LOG(ERROR) << "Dump resource monitor status: " << m_resMonitor.DebugString();

    // Forcely kill one session
    for (auto [usage, pSess] : candidates) {
        // for logging
        forceEvicitedSess = pSess.get()->sessHandle;

        // Don't retry anymore for OOM kernels in this session
        pSess.get()->protectOOM = false;

        VLOG(2) << "Force evict session: " << pSess.get()->sessHandle << " with usage " << usage;
        pSess.get()->interrupt();

        return true;
    }
    LOG(ERROR) << "Nothing to force evict";
    return false;
}

std::unique_ptr<ResourceContext> TaskExecutor::makeResourceContext(PSessionItem sess, uint64_t graphId,
                                                                   const DeviceSpec &spec,
                                                                   const Resources &res, Resources *missing)
{
    auto maybeTicket = m_resMonitor.preAllocate(res, missing);
    if (!maybeTicket) {
        logScheduleFailure(res, m_resMonitor);
        return nullptr;
    }

    auto rctx = std::make_unique<ResourceContext>(m_resMonitor, graphId, spec, *maybeTicket);

#if defined(SALUS_ENABLE_STATIC_STREAM)
    rctx->sessHandle = sess->sessHandle;
#endif

    rctx->addListener(std::move(sess));

    return rctx;
}

} // namespace salus
