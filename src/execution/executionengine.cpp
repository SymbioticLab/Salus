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

#include "executionengine.h"

#include "execution/scheduler/ischeduler.h"
#include "execution/scheduler/sessionitem.h"
#include "execution/scheduler/operationitem.h"
#include "execution/operationtask.h"
#include "platform/logging.h"
#include "utils/containerutils.h"
#include "utils/date.h"
#include "utils/debugging.h"
#include "utils/envutils.h"
#include "utils/macros.h"
#include "utils/pointerutils.h"
#include "utils/threadutils.h"

#include <algorithm>
#include <functional>
#include <iomanip>

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
using FpSeconds = std::chrono::duration<double, seconds::period>;
using namespace std::chrono_literals;
using namespace date;

namespace {
inline void logScheduleFailure(const Resources &usage, const ResourceMonitor &resMon)
{
    UNUSED(usage);
    UNUSED(resMon);

#ifndef NDEBUG
    VLOG(2) << "Try to allocate resource failed. Requested: " << resources::DebugString(usage);
    // Don't call resMon.DebugString directly in log line, as logging acquires lock, and
    // may causing deadlock.
    const auto &str = resMon.DebugString();
    VLOG(2) << "Available: " << str;
#endif
}

} // namespace

ExecutionEngine &ExecutionEngine::instance()
{
    static ExecutionEngine eng;
    return eng;
}

ExecutionEngine::ExecutionEngine()
{
    // Start scheduling thread
    m_schedThread = std::make_unique<std::thread>(std::bind(&ExecutionEngine::scheduleLoop, this));
}

ExecutionEngine::~ExecutionEngine()
{
    // stop scheduling thread
    m_shouldExit = true;
    // also unblock scheduling thread
    m_note_has_work.notify();
    m_schedThread->join();

    // remove any pending new or delete session
    // NOTE: has to be done *after* the scheduling thread exits.
    m_newSessions.clear();
    m_deletedSessions.clear();
}

ExecutionEngine::Inserter ExecutionEngine::registerSession(const std::string &sessHandle)
{
    auto item = std::make_shared<SessionItem>(sessHandle);
    insertSession(item);

    return std::make_shared<InserterImpl>(std::move(item), *this);
}

void ExecutionEngine::insertSession(PSessionItem item)
{
    {
        std::lock_guard<std::mutex> g(m_newMu);
        m_newSessions.emplace_back(std::move(item));
    }
    m_note_has_work.notify();
}

void ExecutionEngine::deleteSession(PSessionItem item)
{
    {
        std::lock_guard<std::mutex> g(m_delMu);
        m_deletedSessions.emplace(std::move(item));
    }
    m_note_has_work.notify();
}

void ExecutionEngine::InserterImpl::enqueueOperation(std::unique_ptr<OperationTask> &&task)
{
    DCHECK(m_item);

    auto opItem = std::make_shared<OperationItem>();
    opItem->sess = m_item;
    opItem->op = std::move(task);
    opItem->tQueued = std::chrono::system_clock::now();

    m_engine.pushToSessionQueue(std::move(opItem));
}

void ExecutionEngine::InserterImpl::registerPagingCallbacks(PagingCallbacks &&pcb)
{
    DCHECK(m_item);
    utils::Guard g(m_item->mu);
    m_item->pagingCb = std::move(pcb);
}

void ExecutionEngine::InserterImpl::deleteSession(std::function<void()> cb)
{
    DCHECK(m_item);

    {
        utils::Guard g(m_item->mu);
        m_item->cleanupCb = std::move(cb);
        // clear paging callbacks so the executorImpl won't get called after it is deleted
        // but haven't been removed from session list yet.
        m_item->pagingCb = {};
    }

    // Request engine to remove session and give up our reference to the session item
    m_engine.deleteSession(std::move(m_item));
}

void ExecutionEngine::pushToSessionQueue(POpItem &&opItem)
{
    auto sess = opItem->sess.lock();
    if (!sess) {
        // session already deleted, discard this task sliently
        return;
    }

    {
        utils::Guard g(sess->mu);
        sess->queue.emplace_back(std::move(opItem));
    }
    m_note_has_work.notify();
}

ExecutionEngine::InserterImpl::~InserterImpl()
{
    if (m_item) {
        m_engine.deleteSession(m_item);
    }
}

bool ExecutionEngine::shouldWaitForAWhile(size_t scheduled, nanoseconds &ns)
{
    static auto last = system_clock::now();
    static auto sleep = 10ms;

    auto now = system_clock::now();

    if (scheduled > 0) {
        last = now;
        sleep = 10ms;
    }

    auto idle = now - last;
    if (idle > 20ms) {
        VLOG(2) << "No progress for " << duration_cast<milliseconds>(idle).count() << "ms, sleep for "
                << duration_cast<milliseconds>(sleep).count() << "ms";
        ns = sleep;
        sleep *= 2;
        return true;
    }
    return false;
}

void ExecutionEngine::scheduleLoop()
{
    m_resMonitor.initializeLimits();
    auto scheduler = SchedulerRegistary::instance().create(m_schedParam.scheduler, *this);
    DCHECK(scheduler);

    m_runningTasks = 0;
    m_noPagingRunningTasks = 0;

    size_t schedIterCount = 0;
    const auto kNameBufLen = 256;
    char schedIterNameBuf[kNameBufLen];
    boost::container::small_vector<PSessionItem, 5> candidates;

    while (!m_shouldExit) {
        snprintf(schedIterNameBuf, kNameBufLen, "sched-iter-%zu", schedIterCount++);
        TIMED_SCOPE(schedIterObj, schedIterNameBuf);

        SessionChangeSet changeset;
        // Fisrt check if there's any pending deletions
        {
            utils::Guard g(m_delMu);

            using std::swap;
            swap(changeset.deletedSessions, m_deletedSessions);
            DCHECK(m_deletedSessions.empty());
        }

        // Delete sessions as requested
        // NOTE: don't clear del yet, we need that in changeset for scheduling
        m_sessions.remove_if([&changeset](auto sess){
            bool deleted = changeset.deletedSessions.count(sess) > 0;
            if (deleted) {
                VLOG(2) << "Deleting session " << sess->sessHandle << "@" << as_hex(sess);
                // The deletion of session's executor is async to this thread.
                // So it's legit for tickets to be nonempty
                // DCHECK(item->tickets.empty());
            }
            return deleted;
        });

        // Append any new sessions
        {
            utils::Guard g(m_newMu);

            changeset.numAddedSessions = m_newSessions.size();

            // list::splice doesn't invalidate iterators, so use
            // m_newSessions.begin() here is ok, and a must.
            changeset.addedSessionBegin = m_newSessions.begin();
            changeset.addedSessionEnd = m_sessions.end();

            m_sessions.splice(m_sessions.end(), m_newSessions);
            DCHECK(m_newSessions.empty());
        }

        // Prepare session ready for this iter of schedule:
        // - move from front end queue to backing storage
        // - reset lastScheduled
        size_t totalRemainingCount = 0;
        bool enableOOMProtect = m_sessions.size() > 1;
        for (auto &item : m_sessions) {
            {
                utils::Guard g(item->mu);
                item->bgQueue.splice(item->bgQueue.end(), item->queue);
            }
            totalRemainingCount += item->bgQueue.size();

            item->protectOOM = enableOOMProtect;
            item->lastScheduled = 0;
        }

        PERFORMANCE_CHECKPOINT_WITH_ID(schedIterObj, "after-accept");

        // Select and sort candidates.
        scheduler->selectCandidateSessions(m_sessions, changeset, &candidates);

        // Deleted sessions are no longer needed, release them.
        changeset.deletedSessions.clear();

        // Schedule tasks from candidate sessions
        // NOTE: remainingCount only counts for candidate sessions in this sched iter.
        size_t remainingCount = 0;
        size_t scheduled = 0;
        for (auto &item : candidates) {
            if (item->forceEvicted) {
                VLOG(2) << "Force evicting pending tasks in session " << item->sessHandle;
                // cancel all pending tasks
                for (auto &opItem : item->bgQueue) {
                    opItem->op->cancel();
                }
                continue;
            }
            VLOG(3) << "Scheduling all opItem in session " << item->sessHandle
                    << ": queue size " << item->bgQueue.size();

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
            << "Scheduler iter stat: " << schedIterCount << " running: " << m_runningTasks
            << " noPageRunning: " << m_noPagingRunningTasks;
        for (auto &item : m_sessions) {
            CLOG(INFO, logging::kPerfTag)
                << "Sched iter " << schedIterCount << " session: " << item->sessHandle
                << " pending: " << item->bgQueue.size() << " scheduled: " << item->lastScheduled
                << " " << scheduler->debugString(item);
        }

        PERFORMANCE_CHECKPOINT_WITH_ID(schedIterObj, "after-sched");

        // Update conditions and check if we need paging
        bool noProgress = remainingCount > 0 && scheduled == 0;
        bool needPaging = (noProgress && m_noPagingRunningTasks == 0);
        if (needPaging && m_sessions.size() > 1) {
            if (doPaging()) {
                // succeed, retry another sched iter immediately
                continue;
            }
        } else if (needPaging) {
            // The single session uses too much memory, we continue without failure retry
            // to let it fail the normal way.
            DCHECK_EQ(m_sessions.size(), 1);
            m_sessions.front()->protectOOM = false;
            continue;
        }

        PERFORMANCE_CHECKPOINT_WITH_ID(schedIterObj, "without-wait");

        std::chrono::nanoseconds ns;
        if (shouldWaitForAWhile(scheduled, ns)) {
            // no progress for a long time.
            // gie out our time slice to avoid using too much cycles
            //             std::this_thread::yield();
            std::this_thread::sleep_for(ns);
        }

        if (!totalRemainingCount) {
            VLOG(2) << "Wait on m_note_has_work";
            m_note_has_work.wait();
        }
    }

    // Cleanup
    m_sessions.clear();
}

std::unique_ptr<ResourceContext> ExecutionEngine::makeResourceContext(SessionItem &sess, const DeviceSpec &spec, const Resources &res)
{
    auto rctx = std::make_unique<ResourceContext>(sess, m_resMonitor);
    if (!rctx->initializeStaging(spec, res)) {
        logScheduleFailure(res, m_resMonitor);
        rctx.reset();
    }
    return rctx;
}

bool ExecutionEngine::maybePreAllocateFor(OperationItem &opItem, const DeviceSpec &spec)
{
    auto item = opItem.sess.lock();
    if (!item) {
        return false;
    }

    auto usage = opItem.op->estimatedUsage(spec);

    auto rctx = makeResourceContext(*item, spec, usage);
    if (!rctx) {
        return false;
    }

    auto ticket = rctx->ticket();
    if (!opItem.op->prepare(std::move(rctx))) {
        return false;
    }

    utils::Guard g(item->tickets_mu);
    item->tickets.insert(ticket);
    return true;
}

POpItem ExecutionEngine::submitTask(POpItem &&opItem)
{
    auto item = opItem->sess.lock();
    if (!item) {
        // discard
        return nullptr;
    }

    opItem->tScheduled = system_clock::now();

    VLOG(3) << "Adding to thread pool: opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();

    // NOTE: this is waited by schedule thread, so we can't afford running
    // the operation inline. If the thread pool is full, simply consider the
    // opItem as not scheduled.

    // opItem has to be captured by value, we need it in case the thread pool is full
    auto c = m_pool.tryRun([opItem, this]() mutable {
        TIMED_SCOPE_IF(timerInnerObj, "ExecutionEngine::maybeScheduleFrom::doSchedule::run", VLOG_IS_ON(1));
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
                    VLOG(2) << "Found expired session during handling of memory failure of opItem: " << opItem->op->DebugString();
                    return false;
                }
                if (!item->protectOOM) {
                    VLOG(2) << "Pass through OOM failed task back to client: " << opItem->op->DebugString();
                    return false;
                }

                taskStopped(*opItem, true);
                // failed due to OOM. Push back to queue and retry later
                VLOG(2) << "Putting back OOM failed task: " << opItem->op->DebugString();
                pushToSessionQueue(std::move(opItem));
                return true;
            };

            VLOG(2) << "Running opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
            taskRunning(*opItem);
            opItem->op->run(cbs);
        }
    });
    if (!c) {
        // successfully sent to thread pool, we can reset opItem
        opItem.reset();
    }
    return opItem;
}

void ExecutionEngine::taskRunning(OperationItem &opItem)
{
    opItem.tRunning = system_clock::now();
    m_runningTasks += 1;
    if (!opItem.op->allowConcurrentPaging()) {
        m_noPagingRunningTasks += 1;
    }
}

void ExecutionEngine::taskStopped(OperationItem &opItem, bool failed)
{
    auto now = system_clock::now();

    auto &rctx = opItem.op->resourceContext();
    rctx.releaseStaging();

    // For now only count memory usage, and simply add up memory usages on different
    // devices.
    if (!failed) {
        CLOG(INFO, logging::kPerfTag)
            << "OpItem Stat "
            << opItem.op->DebugString()
            << " queued: " << opItem.tQueued << " scheduled: " << opItem.tScheduled << " finished: " << now;
    }

    m_runningTasks -= 1;
    if (!opItem.op->allowConcurrentPaging()) {
        m_noPagingRunningTasks -= 1;
    }
}

bool ExecutionEngine::doPaging()
{
    auto now = system_clock::now();
    size_t released = 0;
    std::string forceEvicitedSess;

    utils::ScopeGuards sg([&now, &released, &forceEvicitedSess]() {
        auto dur = system_clock::now() - now;
        CLOG(INFO, logging::kPerfTag)
            << "Paging: "
            << " duration: " << duration_cast<microseconds>(dur).count() << " us"
            << " released: " << released << " forceevict: '" << forceEvicitedSess << "'";
    });

    // Step 1: select candidate sessions
    std::vector<std::pair<size_t, utils::not_null<SessionItem*>>> candidates;
    candidates.reserve(m_sessions.size());

    // Step 1.1: count total memory usage for each session
    // TODO: we currently assume we are paging GPU memory to CPU, make it generic to use Resources
    const static ResourceTag gpuTag{ResourceType::MEMORY, {DeviceType::GPU, 0}};
    const static ResourceTag cpuTag{ResourceType::MEMORY, {DeviceType::CPU, 0}};

    for (auto &pSess : m_sessions) {
        candidates.emplace_back(pSess->resourceUsage(gpuTag), pSess.get());
    }

    // sort in des order
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
            VLOG(2) << "Session " << pSess->sessHandle << " usage: " << usage;
        }
    }

    // Step 2: inform owner to do paging given suggestion
    for (size_t i = 1; i != candidates.size(); ++i) {
        auto pSess = candidates[i].second;
        std::vector<std::pair<size_t, uint64_t>> victims;
        {
            utils::Guard g(pSess->tickets_mu);
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
        utils::Guard g(pSess->mu);
        if (!pSess->pagingCb) {
            continue;
        }

        VLOG(2) << "Visiting session: " << pSess->sessHandle;

        for (auto [usage, victim] : victims) {
            // preallocate some CPU memory for use.
            Resources res{{cpuTag, usage}};

            auto rctx = makeResourceContext(*pSess, devices::CPU0, res);
            if (!rctx) {
                LOG(ERROR) << "No enough CPU memory for paging. Required: " << res[cpuTag] << " bytes";
                return false;
            }
            AllocLog(INFO) << "Pre allocated " << *rctx << " for session=" << pSess->sessHandle;

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
        LOG(ERROR) << "Session " << pSess->sessHandle << " usage: " << usage;
    }
    LOG(ERROR) << "Dump resource monitor status: " << m_resMonitor.DebugString();

    // Forcely kill one session
    for (auto [usage, pSess] : candidates) {
        utils::Guard g(pSess->mu);
        if (!pSess->pagingCb) {
            continue;
        }
        forceEvicitedSess = pSess->sessHandle;

        // Don't retry anymore for OOM kernels in this session
        pSess->protectOOM = false;
        pSess->forceEvicted = true;

        VLOG(2) << "Force evict session: " << pSess->sessHandle << " with usage " << usage;
        pSess->pagingCb.forceEvicted();
        return true;
    }
    LOG(ERROR) << "Nothing to force evict";
    return false;
}

ResourceContext::ResourceContext(const ResourceContext &other, const DeviceSpec &spec)
    : resMon(other.resMon)
    , m_spec(spec)
    , m_ticket(other.m_ticket)
    , session(other.session)
    , hasStaging(false)
{
}

ResourceContext::ResourceContext(SessionItem &item, ResourceMonitor &resMon)
    : resMon(resMon)
    , m_spec()
    , m_ticket(0)
    , session(item)
    , hasStaging(false)
{
}

bool ResourceContext::initializeStaging(const DeviceSpec &spec, const Resources &res)
{
    this->m_spec = spec;
    auto ok = resMon.preAllocate(res, &m_ticket);
    hasStaging = ok;
    return ok;
}

void ResourceContext::releaseStaging()
{
    if (!hasStaging) {
        return;
    }
    resMon.free(m_ticket);
    hasStaging = false;

    // clean up session tickets
    if (!resMon.hasUsage(m_ticket)) {
        removeTicketFromSession();
    }
}

void ResourceContext::removeTicketFromSession() const
{
    // last resource freed
    utils::Guard g(session.tickets_mu);
    VLOG(2) << "Removing ticket " << m_ticket << " from session " << session.sessHandle;
    session.tickets.erase(m_ticket);
}

ResourceContext::~ResourceContext()
{
    releaseStaging();
}

ResourceContext::OperationScope ResourceContext::allocMemory(size_t num_bytes) const
{

    OperationScope scope(*this, resMon.lock());

    scope.res[{ResourceType::MEMORY, m_spec}] = num_bytes;
    scope.valid = scope.proxy.allocate(m_ticket, scope.res);

    return scope;
}

void ResourceContext::deallocMemory(size_t num_bytes) const
{
    ResourceTag tag{ResourceType::MEMORY, m_spec};
    Resources res{{tag, num_bytes}};

    bool ticketIsEmpty = resMon.free(m_ticket, res);
    session.resourceUsage(tag) -= num_bytes;

    if (ticketIsEmpty) {
        removeTicketFromSession();
    }
}

void ResourceContext::OperationScope::rollback()
{
    DCHECK(valid);
    proxy.free(context.ticket(), res);
    // no need to call removeTicketFromSession
    // because this most likely will not be the last deallocation
}

void ResourceContext::OperationScope::commit()
{
    if (!valid) {
        return;
    }

    // the allocation is used by the session (i.e. the session left the scope without rollback)
    for (auto p : res) {
        context.session.resourceUsage(p.first) += p.second;
    }
}

std::ostream &operator<<(std::ostream &os, const ResourceContext &c)
{
    if (c.ticket() == 0) {
        return os << "AllocationTicket(Invalid)";
    }
    return os << "AllocationTicket(" << c.ticket() << ", device=" << c.spec() << ")";
}
