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

#include "execution/resources.h"
#include "execution/operationtask.h"
#include "utils/macros.h"
#include "utils/envutils.h"
#include "utils/threadutils.h"
#include "utils/debugging.h"

#include <functional>

using std::chrono::steady_clock;
using namespace std::chrono_literals;

// #define ENABLE_STACK_SENTINEL

namespace {
void logScheduleFailure(const ResourceMap &usage, const ResourceMonitor &resMon)
{
    STACK_SENTINEL;

    DEBUG("Try to allocate resource failed. Requested: {}", usage.DebugString());
    DEBUG("Available: {}", resMon.DebugString());
}

} // namespace

ExecutionEngine &ExecutionEngine::instance()
{
    static ExecutionEngine eng;
    STACK_SENTINEL;
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
    STACK_SENTINEL;
    // Start scheduling thread
    m_schedThread = std::make_unique<std::thread>(std::bind(&ExecutionEngine::scheduleLoop, this));
}

ExecutionEngine::~ExecutionEngine()
{
    STACK_SENTINEL;
    // stop scheduling thread
    m_shouldExit = true;
    m_schedThread->join();

    // remove any pending new or delete session
    // NOTE: has to be done *after* the scheduling thread exits.
    m_newSessions.clear();
    m_deletedSessions.clear();
}

namespace {
bool useGPU()
{
    STACK_SENTINEL;
    auto use = utils::fromEnvVar("EXEC_SCHED_USE_GPU", true);
    INFO("Scheduling using: {}", use ? "GPU,CPU" : "CPU");
    return use;
}

} // namespace

bool ExecutionEngine::schedule(ITask *t)
{
    STACK_SENTINEL;
    return trySchedule(t, DeviceType::CPU);
}

bool ExecutionEngine::trySchedule(ITask *t, const DeviceSpec &dev)
{
    STACK_SENTINEL;
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
    STACK_SENTINEL;

    auto item = std::make_shared<SessionItem>(sessHandle);
    insertSession(item);

    return std::make_shared<InserterImpl>(std::move(item), *this);
}

void ExecutionEngine::insertSession(std::shared_ptr<SessionItem> item)
{
    STACK_SENTINEL;
    {
        std::lock_guard<std::mutex> g(m_newMu);
        m_newSessions.emplace_back(std::move(item));
    }
    m_note_has_work.notify();
}

void ExecutionEngine::deleteSession(std::shared_ptr<SessionItem> item)
{
    STACK_SENTINEL;
    {
        std::lock_guard<std::mutex> g(m_delMu);
        m_deletedSessions.emplace(std::move(item));
    }
    m_note_has_work.notify();
}

std::future<void> ExecutionEngine::InserterImpl::enqueueOperation(std::unique_ptr<OperationTask> &&task)
{
    STACK_SENTINEL;
    auto opItem = std::make_shared<OperationItem>();
    opItem->op = std::move(task);

    m_engine.pushToSessionQueue(m_item, opItem);

    assert(opItem);

    return opItem->promise.get_future();
}

void ExecutionEngine::pushToSessionQueue(std::shared_ptr<SessionItem> item, std::shared_ptr<OperationItem> opItem)
{
    STACK_SENTINEL;
    {
        utils::Guard g(item->mu);
        item->queue.push_back(opItem);
    }
    m_note_has_work.notify();
}

ExecutionEngine::InserterImpl::~InserterImpl()
{
    STACK_SENTINEL;
    if (m_item)
        m_engine.deleteSession(m_item);
}

bool ExecutionEngine::shouldWaitForAWhile(size_t scheduled, std::chrono::nanoseconds &ns)
{
    STACK_SENTINEL;
    static auto last = steady_clock::now();
    static auto sleep = 10ms;

    auto now = steady_clock::now();

    if (scheduled > 0) {
        last = now;
        sleep = 10ms;
    }

    std::chrono::nanoseconds idle = now - last;
    if (idle > 20ms) {
        INFO("No progress for {}ms, sleep for {}ms",
             std::chrono::duration_cast<std::chrono::milliseconds>(idle).count(),
             std::chrono::duration_cast<std::chrono::milliseconds>(sleep).count());
        ns = sleep;
        sleep *= 2;
        return true;
    }
    return false;
}

void ExecutionEngine::scheduleLoop()
{
    STACK_SENTINEL;
    ResourceMonitor resMon;
    resMon.initializeLimits();

    SessionList sessions;


    while (!m_shouldExit) {
        STACK_SENTINEL;
        TRACE("Scheduler loop");
        // Fisrt check if there's any pending deletions
        SessionSet del;
        {
            utils::Guard g(m_delMu);
            using std::swap;
            swap(del, m_deletedSessions);
            assert(m_deletedSessions.size() == 0);
        }
        TRACE("Got {} session to delete", del.size());

        // Append any new sessions
        {
            utils::Guard g(m_newMu);

            TRACE("Got {} session to add", m_newSessions.size());

            sessions.splice(sessions.end(), m_newSessions);
            assert(m_newSessions.size() == 0);
        }

        TRACE("Handling {} sessions in this iteration", sessions.size());
        // Loop through sessions
        size_t scheduled = 0;
        size_t remainingCount = 0;
        auto it = sessions.begin();
        auto end = sessions.end();
        while (it != end) {
            auto &item = *it;
            if (del.count(item)) {
                TRACE("Deleting session {}@{}", item->sessHandle, as_hex(item));

                resMon.clear(item->sessHandle);
                it = sessions.erase(it);
            } else {
                // Move from front end queue to backing storage
                TRACE("Looking at session {}@{}", item->sessHandle, as_hex(item));
                TRACE("bgQueue has {} opItems", item->bgQueue.size());
                {
                    utils::Guard g(item->mu);
                    item->bgQueue.splice(item->bgQueue.end(), item->queue);
                }
                remainingCount += item->bgQueue.size();
                TRACE("bgQueue has {} opItems after collection", item->bgQueue.size());

                // NOTE: don't use scheduled || maybeScheduleFrom(...)
                // we don't want short-cut eval here and maybeScheduleFrom
                // should always be called.
                auto count = maybeScheduleFrom(resMon, item);
                remainingCount -= count;
                scheduled += count;
                ++it;
            }
        }

        for (auto &d : del) {
            ERR("Session {} requested for deletion but not found in queue", d->sessHandle);
        }

        std::chrono::nanoseconds ns;
        if (shouldWaitForAWhile(scheduled, ns)) {
            // no progress for a long time.
            // gie out our time slice to avoid using too much cycles
//             std::this_thread::yield();
            std::this_thread::sleep_for(ns);
        }

        if (!remainingCount) {
            INFO("Wait on m_note_has_work");
            m_note_has_work.wait();
        }
    }

    // Cleanup
    sessions.clear();
}

ExecutionEngine::SessionItem::~SessionItem()
{
    STACK_SENTINEL;
    bgQueue.clear();
    queue.clear();
}

bool tryScheduleOn(ResourceMonitor &resMon, OperationTask *t, const std::string &sessHandle,
                   const DeviceSpec &dev)
{
    STACK_SENTINEL;
    auto expectedDev = dev;

    auto usage = t->estimatedUsage(expectedDev);
    if (!resMon.tryAllocate(usage, sessHandle)) {
        logScheduleFailure(usage, resMon);
        return false;
    }
    if (t->prepare(expectedDev)) {
        return true;
    }

    resMon.free(usage);
    return false;
}

size_t ExecutionEngine::maybeScheduleFrom(ResourceMonitor &resMon, std::shared_ptr<SessionItem> item)
{
    STACK_SENTINEL;
    auto &queue = item->bgQueue;

    auto size = queue.size();

    TRACE("Scheduling all opItem in session {}: queue size {}", item->sessHandle, size);

    if (size == 0) {
        return 0;
    }

    // Try schedule the operation
    auto doSchedule = [&resMon, this](std::shared_ptr<SessionItem> item, std::shared_ptr<OperationItem> &&opItem) -> std::shared_ptr<OperationItem>{
        STACK_SENTINEL;
        TRACE("Scheduling opItem in session {}: {}", item->sessHandle, opItem->op->DebugString());

        bool scheduled = false;
        DeviceSpec spec;
        for (auto dt : opItem->op->supportedDeviceTypes()) {
            if (dt == DeviceType::GPU && !useGPU()) {
                continue;
            }
            spec = DeviceSpec(dt, 0);
            if (tryScheduleOn(resMon, opItem->op.get(), item->sessHandle, spec)) {
                INFO("Task scheduled on {}", spec.DebugString());
                scheduled = true;
                break;
            }
        }

        // Send to thread pool
        if (scheduled) {
            TRACE("Adding to thread pool: opItem in session {}: {}", item->sessHandle, opItem->op->DebugString());
            q::with(m_qec->queue(), std::move(opItem)).then([&resMon, spec, item, this](std::shared_ptr<OperationItem> &&opItem){
                STACK_SENTINEL;
                OperationTask::Callbacks cbs;

                assert(item);
                assert(opItem);

                cbs.launched = [opItem]() {
                    opItem->promise.set_value();
                };
                cbs.done = [&resMon, spec, opItem]() {
                    // succeed
                    ResourceMap res;
                    opItem->op->lastUsage(spec, res);
                    resMon.free(res);
                };
                cbs.memFailure = [&resMon, spec, opItem, item, this]() mutable {
                    // failed due to OOM. Push back to queue
                    WARN("Opkernel {} failed due to OOM", opItem->op->DebugString());
                    ResourceMap res;
                    opItem->op->lastUsage(spec, res);
                    resMon.free(res);
                    pushToSessionQueue(item, std::move(opItem));
                };

                TRACE("Running opItem in session {}: {}", item->sessHandle, opItem->op->DebugString());
                opItem->op->run(cbs);
            });
        } else {
            TRACE("Failed to schedule opItem in session {}: {}", item->sessHandle, opItem->op->DebugString());
        }
        return opItem;;
    };

    // Do all schedule in queue in parallel
    UnsafeQueue stage;
    stage.swap(queue);
    std::vector<q::promise<std::shared_ptr<OperationItem>>> promises;
    for (auto &opItem : stage) {
        auto p = q::with(m_qec->queue(), item, std::move(opItem)).then(doSchedule);
        promises.emplace_back(std::move(p));
    }

    assert(queue.empty());
    TRACE("All opItem in session {} exaimed", item->sessHandle);

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

    TRACE("Adding back {} opItem in session {}", queue.size(), item->sessHandle);

    return size - queue.size();
}
