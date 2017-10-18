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
#include "utils/macros.h"
#include "utils/envutils.h"
#include "utils/threadutils.h"
#include "utils/containerutils.h"
#include "utils/debugging.h"

#include <functional>
#include <algorithm>

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using namespace std::chrono_literals;

namespace {
inline void logScheduleFailure(const Resources &usage, const ResourceMonitor &resMon)
{
    TIMED_FUNC(timerObj);

    UNUSED(usage);
    UNUSED(resMon);

#ifndef NDEBUG
    VLOG(1) << "Try to allocate resource failed. Requested: " << resources::DebugString(usage);
    // Don't call resMon.DebugString directly in log line, as logging acquires lock, and
    // may causing deadlock.
    const auto &str = resMon.DebugString();
    VLOG(1) << "Available: " << str;
#endif
}

} // namespace

ExecutionEngine &ExecutionEngine::instance()
{
    static ExecutionEngine eng;
    TIMED_FUNC(timerObj);
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
    TIMED_FUNC(timerObj);
    // Start scheduling thread
    m_schedThread = std::make_unique<std::thread>(std::bind(&ExecutionEngine::scheduleLoop, this));
}

ExecutionEngine::~ExecutionEngine()
{
    TIMED_FUNC(timerObj);
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
    TIMED_FUNC(timerObj);
    auto use = utils::fromEnvVar("EXEC_SCHED_USE_GPU", true);
    VLOG(1) << "Scheduling using: " << (use ? "GPU,CPU" : "CPU");
    return use;
}

} // namespace

bool ExecutionEngine::schedule(ITask *t)
{
    TIMED_FUNC(timerObj);
    return trySchedule(t, DeviceType::CPU);
}

bool ExecutionEngine::trySchedule(ITask *t, const DeviceSpec &dev)
{
    TIMED_FUNC(timerObj);
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
    TIMED_FUNC(timerObj);

    auto item = std::make_shared<SessionItem>(sessHandle);
    insertSession(item);

    return std::make_shared<InserterImpl>(std::move(item), *this);
}

void ExecutionEngine::insertSession(std::shared_ptr<SessionItem> item)
{
    TIMED_FUNC(timerObj);
    {
        std::lock_guard<std::mutex> g(m_newMu);
        m_newSessions.emplace_back(std::move(item));
    }
    m_note_has_work.notify();
}

void ExecutionEngine::deleteSession(std::shared_ptr<SessionItem> item)
{
    TIMED_FUNC(timerObj);
    {
        std::lock_guard<std::mutex> g(m_delMu);
        m_deletedSessions.emplace(std::move(item));
    }
    m_note_has_work.notify();
}

void ExecutionEngine::InserterImpl::enqueueOperation(std::unique_ptr<OperationTask> &&task)
{
    TIMED_FUNC(timerObj);
    auto opItem = std::make_shared<OperationItem>();
    opItem->op = std::move(task);
    opItem->tQueued = std::chrono::steady_clock::now();

    m_engine.pushToSessionQueue(m_item, std::move(opItem));
}

void ExecutionEngine::InserterImpl::registerPagingCallbacks(PagingCallbacks &&pcb)
{
    TIMED_FUNC(timerObj);

    utils::Guard g(m_item->mu);
    m_item->pagingCb = std::move(pcb);
}

void ExecutionEngine::pushToSessionQueue(std::shared_ptr<SessionItem> item, std::shared_ptr<OperationItem> opItem)
{
    TIMED_FUNC(timerObj);
    {
        utils::Guard g(item->mu);
        item->queue.push_back(opItem);
    }
    m_note_has_work.notify();
}

ExecutionEngine::InserterImpl::~InserterImpl()
{
    TIMED_FUNC(timerObj);
    if (m_item)
        m_engine.deleteSession(m_item);
}

bool ExecutionEngine::shouldWaitForAWhile(size_t scheduled, nanoseconds &ns)
{
    TIMED_FUNC(timerObj);
    static auto last = steady_clock::now();
    static auto sleep = 10ms;

    auto now = steady_clock::now();

    if (scheduled > 0) {
        last = now;
        sleep = 10ms;
    }

    auto idle = now - last;
    if (idle > 20ms) {
        VLOG(1) << "No progress for " << duration_cast<milliseconds>(idle).count()
                << "ms, sleep for " << duration_cast<milliseconds>(sleep).count() << "ms";
        ns = sleep;
        sleep *= 2;
        return true;
    }
    return false;
}

void ExecutionEngine::scheduleLoop()
{
    TIMED_FUNC(timerObj);
    m_resMonitor.initializeLimits();

    m_runningTasks = 0;
    m_noPagingRunningTasks = 0;

    while (!m_shouldExit) {
        TIMED_SCOPE(schedIterObj, "sched-iter");

        int sessionsChanged = 0;
        // Fisrt check if there's any pending deletions
        SessionSet del;
        {
            utils::Guard g(m_delMu);

            sessionsChanged += m_deletedSessions.size();

            using std::swap;
            swap(del, m_deletedSessions);
            DCHECK(m_deletedSessions.size() == 0);
        }

        // Append any new sessions
        {
            utils::Guard g(m_newMu);

            sessionsChanged += m_newSessions.size();

            m_sessions.splice(m_sessions.end(), m_newSessions);
            DCHECK(m_newSessions.size() == 0);
        }

        // Snapshot resource usage counter first, or reset them
        // and delete sessions as requested
        for (auto it = m_sessions.begin(),
             itend = m_sessions.end(); it != itend;) {
            auto &item = *it;
            if (del.erase(item) > 0) {
                VLOG(2) << "Deleting session " << item->sessHandle << "@" << as_hex(item);
                DCHECK(item.use_count() == 1);
                // The deletion of session's executor is async to this thread.
                // So it's legit for tickets to be nonempt
                // DCHECK(item->tickets.empty());
                it = m_sessions.erase(it);
            } else {
                if (sessionsChanged == 0) {
                    item->unifiedResSnapshot = item->unifiedRes;
                } else {
                    item->unifiedResSnapshot = item->unifiedRes = 0;
                }
                ++it;
            }
        }

        PERFORMANCE_CHECKPOINT_WITH_ID(schedIterObj, "after-snapshot");

        // Sort sessions if needed. We assume m_sessions.size() is always no more than a few,
        // therefore sorting in every iteration is acceptable.
        if (sessionsChanged == 0) {
            m_sessions.sort([](const auto &lhs, const auto &rhs){
                return lhs->unifiedResSnapshot < rhs->unifiedResSnapshot;
            });
        }

        if (VLOG_IS_ON(2)) {
            for (auto &sess : m_sessions) {
                VLOG(2) << "Progress counter for session " << sess->sessHandle << ": " << sess->unifiedResSnapshot;
            }
        }

        // Loop through and accept new tasks
        size_t remainingCount = 0;
        for (auto &item : m_sessions) {
            // Move from front end queue to backing storage
            {
                utils::Guard g(item->mu);
                item->bgQueue.splice(item->bgQueue.end(), item->queue);
            }
            remainingCount += item->bgQueue.size();
        }

        // Schedule in order
        size_t scheduled = 0;
        for (auto &item : m_sessions) {
            auto count = maybeScheduleFrom(item);
            scheduled += count;
            remainingCount -= count;
            // make sure the first session (with least progress) is
            // get scheduled solely, thus can keep up, without other
            // sessions interfere
            if (count > 0) {
                break;
            }
        }

        PERFORMANCE_CHECKPOINT_WITH_ID(schedIterObj, "after-schedule");

        // Update conditions and check if we need paging
        bool noProgress = remainingCount > 0 && scheduled == 0;
        int64_t running_tasks = m_noPagingRunningTasks;
        bool needPaging = (noProgress && running_tasks == 0);
        if (needPaging) {
            TIMED_SCOPE(pagingObj, "paging");
            if (doPaging()) {
                // succeed, retry immediately
                continue;
            }
        }

        std::chrono::nanoseconds ns;
        if (shouldWaitForAWhile(scheduled, ns)) {
            // no progress for a long time.
            // gie out our time slice to avoid using too much cycles
//             std::this_thread::yield();
            std::this_thread::sleep_for(ns);
        }

        if (!remainingCount) {
            VLOG(1) << "Wait on m_note_has_work";
            m_note_has_work.wait();
        }
    }

    // Cleanup
    m_sessions.clear();
}

ExecutionEngine::SessionItem::~SessionItem()
{
    TIMED_FUNC(timerObj);
    bgQueue.clear();
    queue.clear();
}

bool ExecutionEngine::maybePreAllocateFor(SessionItem &item, OperationItem &opItem, const DeviceSpec &spec)
{
    TIMED_FUNC(timerObj);

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

size_t ExecutionEngine::maybeScheduleFrom(std::shared_ptr<SessionItem> item)
{
    TIMED_FUNC(timerObj);

    auto &queue = item->bgQueue;

    auto size = queue.size();

    VLOG(3) << "Scheduling all opItem in session " << item->sessHandle << ": queue size " << size;

    if (size == 0) {
        return 0;
    }

    // Try schedule the operation
    auto doSchedule = [this](std::shared_ptr<SessionItem> item, std::shared_ptr<OperationItem> &&opItem) {
        VLOG(3) << "Scheduling opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
        TIMED_SCOPE(timerInnerObj, "ExecutionEngine::maybeScheduleFrom::doSchedule");

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
            opItem->tScheduled = steady_clock::now();

            VLOG(3) << "Adding to thread pool: opItem in session " << item->sessHandle
                    << ": " << opItem->op->DebugString();
            q::with(m_qec->queue(), std::move(opItem)).then([item, this](std::shared_ptr<OperationItem> &&opItem){
                TIMED_SCOPE(timerInnerObj, "ExecutionEngine::maybeScheduleFrom::doSchedule::run");
                OperationTask::Callbacks cbs;

                DCHECK(item);
                DCHECK(opItem);

                cbs.done = [item, opItem, this]() {
                    // succeed
                    VLOG(2) << "OpItem " << opItem->op->DebugString() << " queuing time: "
                            << duration_cast<milliseconds>(opItem->tScheduled - opItem->tQueued).count()
                            << "ms";
                    taskStopped(*item, *opItem);
                };
                cbs.memFailure = [item, opItem, this]() mutable {
                    taskStopped(*item, *opItem);
                    // failed due to OOM. Push back to queue
                    VLOG(1) << "Puting back OOM failed task: " << opItem->op->DebugString();
                    pushToSessionQueue(item, std::move(opItem));
                };

                VLOG(2) << "Running opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
                opItem->op->run(cbs);
            });
        } else {
            VLOG(2) << "Failed to schedule opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
        }
        return opItem;
    };

    // Do all schedule in queue in parallel
    UnsafeQueue stage;
    stage.swap(queue);
    std::vector<q::promise<std::shared_ptr<OperationItem>>> promises;
    for (auto &opItem : stage) {
        auto p = q::with(m_qec->queue(), item, std::move(opItem)).then(doSchedule);
        promises.emplace_back(std::move(p));
    }

    DCHECK(queue.empty());
    VLOG(2) << "All opItem in session " << item->sessHandle << " exaimed";

    auto it = std::back_inserter(queue);
    utils::notification n;
    q::all(std::move(promises), m_qec->queue())
    .then([it, &n](std::vector<std::shared_ptr<OperationItem>> &&remain) mutable {
        for (auto &poi : remain) {
            if (poi) {
                it = std::move(poi);
            }
        }
        n.notify();
    });
    n.wait();

    return size - queue.size();
}

void ExecutionEngine::taskStopped(SessionItem &item, OperationItem &opItem)
{
    UNUSED(item);

    auto &rctx = opItem.op->resourceContext();
    rctx.releaseStaging();

    auto dur = duration_cast<microseconds>(steady_clock::now() - opItem.tScheduled).count();

    // For now only count memory usage, and simply add up memory usages on different
    // devices.
    // TODO: find better formula to do this
    auto memUsage = m_resMonitor.queryUsage(rctx.ticket());
    uint64_t unifiedRes = 0;
    if (memUsage) {
        for (auto &p : *memUsage) {
            if (p.first.type != ResourceType::MEMORY) {
                continue;
            }
            unifiedRes += p.second;
        }
    } else {
        unifiedRes = 1;
    }
    unifiedRes *= dur;
    item.unifiedRes += unifiedRes;

    m_runningTasks -= 1;
    if (!opItem.op->allowConcurrentPaging()) {
        m_noPagingRunningTasks -= 1;
    }
}

bool ExecutionEngine::doPaging()
{
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
        auto usages = m_resMonitor.queryUsages(pSess->tickets);
        auto mem = utils::getOrDefault(usages, gpuTag, 0);
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
        LOG(ERROR) << "No candidates to do paging";
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
            auto released = sess.pagingCb.volunteer(victim, std::move(rctx));
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

    return false;
    // Step 3: TODO: force evict
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

    OperationScope scope(resMon.lock());

    scope.res[{ResourceType::MEMORY, m_spec}] = num_bytes;
    scope.ticket = m_ticket;
    scope.valid = scope.proxy.allocate(m_ticket, scope.res);

    return scope;
}

void ResourceContext::deallocMemory(size_t num_bytes) const
{
    Resources res{
        {{ResourceType::MEMORY, m_spec}, num_bytes}
    };

    if (resMon.free(m_ticket, res)) {
        removeTicketFromSession();
    }
}

void ResourceContext::OperationScope::rollback()
{
    DCHECK(valid);
    proxy.free(ticket, res);
    // no need to call removeTicketFromSession
    // because this most likely will not be the last deallocation
}

std::ostream &operator<<(std::ostream &os, const ResourceContext &c)
{
    if (c.ticket() == 0) {
        return os << "AllocationTicket(Invalid)";
    }
    return os << "AllocationTicket(" << c.ticket() << ", device=" << c.spec() << ")";
}
