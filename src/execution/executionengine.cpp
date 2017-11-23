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

#include "execution/operationtask.h"
#include "platform/logging.h"
#include "utils/macros.h"
#include "utils/envutils.h"
#include "utils/threadutils.h"
#include "utils/containerutils.h"
#include "utils/date.h"
#include "utils/debugging.h"

#include <functional>
#include <algorithm>
#include <iomanip>

using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::seconds;
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
    : m_qscope(q::scoped_initialize())
    , m_qec(q::make_execution_context<q::threadpool,
                                      q::direct_scheduler>("executionengine",
                                                           // The queue passed in here is only used for threadpool
                                                           // termination. We don't care about it. Thus this queue
                                                           // is not connected to any event dispatcher
                                                           q::make_shared<q::queue>(0)))
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

namespace {
bool useGPU()
{
    auto use = utils::fromEnvVar("EXEC_SCHED_USE_GPU", true);
    VLOG(2) << "Scheduling using: " << (use ? "GPU,CPU" : "CPU");
    return use;
}

} // namespace

bool ExecutionEngine::schedule(ITask *t)
{
    return trySchedule(t, DeviceType::CPU);
}

bool ExecutionEngine::trySchedule(ITask *t, const DeviceSpec &dev)
{
    auto expectedDev = dev;
    if (t->prepare(expectedDev)) {
        return true;
    }

    if (expectedDev != dev) {
        // the task wants to run on a different device
        return t->prepare(expectedDev);
    }
    return false;
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
    auto opItem = std::make_shared<OperationItem>();
    opItem->op = std::move(task);
    opItem->tQueued = std::chrono::system_clock::now();

    m_engine.pushToSessionQueue(m_item, std::move(opItem));
}

void ExecutionEngine::InserterImpl::registerPagingCallbacks(PagingCallbacks &&pcb)
{
    utils::Guard g(m_item->mu);
    m_item->pagingCb = std::move(pcb);
}

void ExecutionEngine::InserterImpl::deleteSession(std::function<void()> cb)
{
    {
        utils::Guard g(m_item->mu);
        m_item->cleanupCb = std::move(cb);
    }
    m_engine.deleteSession(std::move(m_item));
}

void ExecutionEngine::pushToSessionQueue(PSessionItem item, POpItem opItem)
{
    {
        utils::Guard g(item->mu);
        item->queue.push_back(opItem);
    }
    m_note_has_work.notify();
}

ExecutionEngine::InserterImpl::~InserterImpl()
{
    if (m_item)
        m_engine.deleteSession(m_item);
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
        VLOG(2) << "No progress for " << duration_cast<milliseconds>(idle).count()
                << "ms, sleep for " << duration_cast<milliseconds>(sleep).count() << "ms";
        ns = sleep;
        sleep *= 2;
        return true;
    }
    return false;
}

void ExecutionEngine::scheduleLoop()
{
    m_resMonitor.initializeLimits();

    m_runningTasks = 0;
    m_noPagingRunningTasks = 0;

    size_t schedIterCount = 0;
    const auto kNameBufLen = 256;
    char schedIterNameBuf[kNameBufLen];

    auto lastSnapshotTime = system_clock::now();

    while (!m_shouldExit) {
        snprintf(schedIterNameBuf, kNameBufLen, "sched-iter-%zu", schedIterCount++);
        TIMED_SCOPE(schedIterObj, schedIterNameBuf);

        // Fisrt check if there's any pending deletions
        SessionSet del;
        {
            utils::Guard g(m_delMu);

            using std::swap;
            swap(del, m_deletedSessions);
            DCHECK(m_deletedSessions.size() == 0);
        }

        // Append any new sessions
        int sessionsChanged = 0;
        {
            utils::Guard g(m_newMu);

            sessionsChanged += m_newSessions.size();

            m_sessions.splice(m_sessions.end(), m_newSessions);
            DCHECK(m_newSessions.size() == 0);
        }

        PERFORMANCE_CHECKPOINT_WITH_ID(schedIterObj, "after-accept");
        // Snapshot resource usage counter first, or reset them
        // and delete sessions as requested
        auto now = system_clock::now();
        auto sSinceLastSnapshot = FpSeconds(now - lastSnapshotTime).count();
        lastSnapshotTime = now;
        for (auto it = m_sessions.begin(),
             itend = m_sessions.end(); it != itend;) {
            auto &item = *it;
            if (del.erase(item) > 0) {
                VLOG(2) << "Deleting session " << item->sessHandle << "@" << as_hex(item);
                DCHECK(item.use_count() == 1);
                // The deletion of session's executor is async to this thread.
                // So it's legit for tickets to be nonempty
                // DCHECK(item->tickets.empty());
                it = m_sessions.erase(it);
            } else {
                if (sessionsChanged == 0) {
                    // calculate progress counter increase since last snapshot
                    size_t mem = item->resourceUsage(ResourceTag::GPU0Memory());
                    item->unifiedResSnapshot += mem * sSinceLastSnapshot;
                } else {
                    item->unifiedResSnapshot = 0;
                }
                ++it;
            }
        }

        PERFORMANCE_CHECKPOINT_WITH_ID(schedIterObj, "after-snapshot");
        // Sort sessions if needed. We assume m_sessions.size() is always no more than a few,
        // therefore sorting in every iteration is acceptable.
        if (sessionsChanged == 0 && m_schedParam.useFairnessCounter) {
            m_sessions.sort([](const auto &lhs, const auto &rhs){
                return lhs->unifiedResSnapshot < rhs->unifiedResSnapshot;
            });
        }
        PERFORMANCE_CHECKPOINT_WITH_ID(schedIterObj, "after-sort");

        // Schedule in order
        size_t totalRemainingCount = 0;
        size_t remainingCount = 0;
        size_t scheduled = 0;
        bool shouldSchedule = true;
        for (auto &item : m_sessions) {
            // Move from front end queue to backing storage
            {
                utils::Guard g(item->mu);
                item->bgQueue.splice(item->bgQueue.end(), item->queue);
            }

            // Try schedule from this session
            size_t count = 0;
            if (shouldSchedule) {
                count = maybeScheduleFrom(item);
                scheduled += count;

                // remaining count is only counted
                // for sessions that are considered for
                // scheduling in this sched iter.
                remainingCount += item->bgQueue.size();
            }
            totalRemainingCount += item->bgQueue.size();

            CLOG(INFO, logging::kPerfTag) << "Sched iter " << schedIterCount
                                          << " session: " << item->sessHandle
                                          << " pending: " << item->bgQueue.size()
                                          << " scheduled: " << count
                                          << " counter: " << item->unifiedResSnapshot;
            if (m_schedParam.useFairnessCounter) {
                // make sure the first session (with least progress) is
                // get scheduled solely, thus can keep up, without other
                // sessions interfere
                if (!m_schedParam.workConservative || count > 0) {
                    shouldSchedule = false;
                }
            }
        }
        CLOG(INFO, logging::kPerfTag) << "Scheduler iter stat: " << schedIterCount
                                      << " running: " << m_runningTasks
                                      << " noPageRunning: " << m_noPagingRunningTasks;

        PERFORMANCE_CHECKPOINT_WITH_ID(schedIterObj, "after-sched");
        // Update conditions and check if we need paging
        bool noProgress = remainingCount > 0 && scheduled == 0;
        int64_t running_tasks = m_noPagingRunningTasks;
        bool needPaging = (noProgress && running_tasks == 0);
        if (needPaging && m_sessions.size() > 1) {
            if (doPaging()) {
                // succeed, retry immediately
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

ExecutionEngine::SessionItem::~SessionItem()
{
    bgQueue.clear();
    queue.clear();

    std::function<void()> cb;
    {
        utils::Guard g(mu);
        cb = std::move(cleanupCb);
    }
    if (cb) {
        cb();
    }
}

bool ExecutionEngine::maybePreAllocateFor(SessionItem &item, OperationItem &opItem, const DeviceSpec &spec)
{
    auto usage = opItem.op->estimatedUsage(spec);

    auto rctx = std::make_unique<ResourceContext>(item, m_resMonitor);
    if (!rctx->initializeStaging(spec, usage)) {
        logScheduleFailure(usage, m_resMonitor);
        return false;
    }

    auto ticket = rctx->ticket();
    if (!opItem.op->prepare(std::move(rctx))) {
        return false;
    }

    utils::Guard g(item.tickets_mu);
    item.tickets.insert(ticket);
    return true;
}

size_t ExecutionEngine::maybeScheduleFrom(PSessionItem item)
{
    auto &queue = item->bgQueue;

    auto size = queue.size();

    VLOG(3) << "Scheduling all opItem in session " << item->sessHandle << ": queue size " << size;

    if (size == 0) {
        return 0;
    }

    // Capture the value in schedule thread, avoid multiple threads accessing this
    bool cancelled = item->forceEvicted;

    // Try schedule the operation
    auto doSchedule = [this, cancelled](PSessionItem item, POpItem &&opItem) -> POpItem {
        if (cancelled) {
            opItem->op->cancel();
            return nullptr;
        }

        VLOG(3) << "Scheduling opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
        TIMED_SCOPE_IF(timerInnerObj, "ExecutionEngine::maybeScheduleFrom::doSchedule", VLOG_IS_ON(1));

        opItem->tInspected = system_clock::now();
        bool scheduled = false;
        DeviceSpec spec;
        for (auto dt : opItem->op->supportedDeviceTypes()) {
            if (dt == DeviceType::GPU && !useGPU()) {
                continue;
            }
            spec = DeviceSpec(dt, 0);
            if (maybePreAllocateFor(*item, *opItem, spec)) {
                VLOG(3) << "Task scheduled on " << spec.DebugString();
                scheduled = true;
                break;
            }
        }

        // Send to thread pool
        if (scheduled) {
            m_runningTasks += 1;
            if (!opItem->op->allowConcurrentPaging()) {
                m_noPagingRunningTasks += 1;
            }
            opItem->tScheduled = system_clock::now();

            VLOG(3) << "Adding to thread pool: opItem in session " << item->sessHandle
                    << ": " << opItem->op->DebugString();
            q::with(m_qec->queue(), std::move(opItem)).then([item, this](POpItem &&opItem){
                TIMED_SCOPE_IF(timerInnerObj, "ExecutionEngine::maybeScheduleFrom::doSchedule::run",
                               VLOG_IS_ON(1));
                OperationTask::Callbacks cbs;

                DCHECK(item);
                DCHECK(opItem);

                cbs.done = [item, opItem, this]() {
                    // succeed
                    taskStopped(*item, *opItem, false);
                };
                cbs.memFailure = [item, opItem, this]() mutable {
                    if (!item->protectOOM) {
                        return false;
                    }

                    taskStopped(*item, *opItem, true);
                    // failed due to OOM. Push back to queue and retry later
                    VLOG(2) << "Puting back OOM failed task: " << opItem->op->DebugString();
                    pushToSessionQueue(item, std::move(opItem));
                    return true;
                };

                if (m_schedParam.randomizedExecution) {
                    milliseconds dur {std::rand() % 100};
                    std::this_thread::sleep_for(dur);
                }
                VLOG(2) << "Running opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
                opItem->tRunning = system_clock::now();
                opItem->op->run(cbs);
            });
        } else {
            VLOG(2) << "Failed to schedule opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
        }
        return opItem;
    };

    // Exam if queue front has been waiting for a long time
    int scheduled = 0;

    if (item->holWaiting > m_schedParam.maxHolWaiting) {
        VLOG(2) << "In session " << item->sessHandle
                << ": HOL waiting exceeds maximum: " << item->holWaiting << " (max="
                << m_schedParam.maxHolWaiting << ")";
        // Only try to schedule head in this case
        auto &head = queue.front();
        head = doSchedule(item, std::move(head));
        if (!head) {
            queue.pop_front();
            scheduled += 1;
        }
    } else {
        // Do all schedule in queue in parallel
        UnsafeQueue stage;
        stage.swap(queue);

        std::vector<q::promise<POpItem>> promises;
        for (auto &opItem : stage) {
            auto p = q::with(m_qec->queue(), item, std::move(opItem)).then(doSchedule);
            promises.emplace_back(std::move(p));
        }

        VLOG(2) << "All opItem in session " << item->sessHandle << " exaimed";

        auto it = std::back_inserter(queue);
        utils::notification n;
        q::all(std::move(promises), m_qec->queue())
        .then([it, &n](std::vector<POpItem> &&remain) mutable {
            for (auto &poi : remain) {
                if (poi) {
                    it = std::move(poi);
                }
            }
            n.notify();
        });
        n.wait();

        scheduled = size - queue.size();
    }

    // update queue head waiting
    if (queue.empty()) {
        item->queueHeadHash = 0;
        item->holWaiting = 0;
    } else if (queue.front()->hash() == item->queueHeadHash) {
        item->holWaiting += scheduled;
    } else {
        item->queueHeadHash = queue.front()->hash();
        item->holWaiting = 0;
    }

    return scheduled;
}

void ExecutionEngine::taskStopped(SessionItem &item, OperationItem &opItem, bool failed)
{
    UNUSED(item);
    auto now = system_clock::now();

    auto &rctx = opItem.op->resourceContext();
    rctx.releaseStaging();

    // For now only count memory usage, and simply add up memory usages on different
    // devices.
    if (!failed) {
        CLOG(INFO, logging::kPerfTag) << "OpItem Stat " << opItem.op->DebugString()
//                                         << " memusage: " << unifiedRes
                                        << " queued: " << opItem.tQueued
                                        << " scheduled: " << opItem.tScheduled
                                        << " finished: " << now;
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
        CLOG(INFO, logging::kPerfTag) << "Paging: "
                                      << " duration: " << duration_cast<microseconds>(dur).count() << " us"
                                      << " released: " << released
                                      << " forceevict: '" << forceEvicitedSess << "'";
    });

    // Step 1: select candidate sessions
    std::vector<std::pair<
        size_t,
        std::reference_wrapper<SessionItem>
    >> candidates;
    candidates.reserve(m_sessions.size());

    // Step 1.1: count total memory usage for each session
    // TODO: we currently assume we are paging GPU memory to CPU, make it generic to use Resources
    ResourceTag gpuTag {ResourceType::MEMORY, {DeviceType::GPU, 0}};
    ResourceTag cpuTag {ResourceType::MEMORY, {DeviceType::CPU, 0}};

    for (auto &pSess : m_sessions) {
        size_t mem = 0;
        {
            utils::Guard g(pSess->tickets_mu);
            auto usages = m_resMonitor.queryUsages(pSess->tickets);
            mem = utils::getOrDefault(usages, gpuTag, 0);
        }
        candidates.emplace_back(mem, *pSess);
    }

    // sort in des order
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &lhs, const auto &rhs){
        return lhs.first > rhs.first;
    });

    // Step 1.2: keep the session with largest memory usage, and try from next
    // no need to erase the first elem, as it's a O(n) operation on vector

    if (candidates.size() <= 1) {
        LOG(ERROR) << "Out of memory for one session";
        return false;
    }

    if (VLOG_IS_ON(2)) {
        for (size_t i = 0; i != candidates.size(); ++i) {
            auto usage = candidates[i].first;
            SessionItem &sess = candidates[i].second;
            VLOG(2) << "Session " << sess.sessHandle << " usage: " << usage;
        }
    }

    // Step 2: inform owner to do paging given suggestion
    for (size_t i = 1; i != candidates.size(); ++i) {
        SessionItem &sess = candidates[i].second;
        std::vector<std::pair<size_t, uint64_t>> victims;
        {
            utils::Guard g(sess.tickets_mu);
            if (sess.tickets.empty()) {
                // no need to go beyond
                break;
            }
            victims = m_resMonitor.sortVictim(sess.tickets);
        }

        // we will be doing paging on this session. Lock it's input queue lock
        // also prevents the executor from clearing the paging callbacks.
        // This should not create deadlock as nothing could finish at this time,
        // thus no new tasks could be submitted.
        utils::Guard g(sess.mu);
        if (!sess.pagingCb) {
            continue;
        }

        VLOG(2) << "Visiting session: " << sess.sessHandle;

        for (auto &p : victims) {
            auto usage = p.first;
            auto victim = p.second;
            // preallocate some CPU memory for use.
            Resources res {
                {cpuTag, usage}
            };

            auto rctx = std::make_unique<ResourceContext>(sess, m_resMonitor);
            if (!rctx->initializeStaging({DeviceType::CPU, 0}, res)) {
                LOG(ERROR) << "No enough CPU memory for paging. Required: " << res[cpuTag] << " bytes";
                return false;
            }
            AllocLog(INFO) << "Pre allocated " << *rctx << " for session=" << sess.sessHandle;

            VLOG(2) << "    request to page out ticket " << victim << " of usage " << usage;
            // request the session to do paging
            released += sess.pagingCb.volunteer(victim, std::move(rctx));
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
    for (size_t i = 0; i != candidates.size(); ++i) {
        auto usage = candidates[i].first;
        SessionItem &sess = candidates[i].second;
        LOG(ERROR) << "Session " << sess.sessHandle << " usage: " << usage;
    }
    LOG(ERROR) << "Dump resource monitor status: " << m_resMonitor.DebugString();

    // Forcely kill one session
    for (size_t i = 1; i != candidates.size(); ++i) {
        SessionItem &sess = candidates[i].second;

        utils::Guard g(sess.mu);
        if (!sess.pagingCb) {
            continue;
        }
        forceEvicitedSess = sess.sessHandle;

        // Don't retry anymore for OOM kernels in this session
        sess.protectOOM = false;
        sess.forceEvicted = true;

        VLOG(2) << "Force evict session: " << sess.sessHandle;
        sess.pagingCb.forceEvicted();
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

ResourceContext::ResourceContext(ExecutionEngine::SessionItem &item, ResourceMonitor& resMon)
    : resMon(resMon)
    , m_ticket(0)
    , session(item)
    , hasStaging(false)
{
}

bool ResourceContext::initializeStaging(const DeviceSpec& spec, const Resources& res)
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
    ResourceTag tag {ResourceType::MEMORY, m_spec};
    Resources res{
        {tag, num_bytes}
    };

    if (resMon.free(m_ticket, res)) {
        session.resourceUsage(tag) -= num_bytes;

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
    if (!valid) return;

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
