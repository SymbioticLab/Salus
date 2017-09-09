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

#include <functional>
#include <chrono>

using std::chrono::steady_clock;
using namespace std::chrono_literals;

namespace {
void logScheduleFailure(const ResourceMap &usage, const ResourceMonitor &resMon)
{
    DEBUG("Try to allocate resource failed. Requested:");
    for (auto p : usage) {
        DEBUG("    {} -> {}", p.first.DebugString(), p.second);
    }
    DEBUG("Available: {}", resMon.DebugString());
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
    m_schedThread->join();

    // remove any pending new or delete session
    // NOTE: has to be done *after* the scheduling thread exits.
    for (auto item : m_newSessions) {
        delete item;
    }
    for (auto item : m_deletedSessions) {
        delete item;
    }
}

namespace {
bool useGPU()
{
    auto use = utils::fromEnvVar("EXEC_SCHED_USE_GPU", true);
    INFO("Scheduling using: {}", use ? "GPU,CPU" : "CPU");
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
    auto item = new SessionItem(sessHandle);

    insertSession(item);

    return std::make_shared<InserterImpl>(item, this);
}

void ExecutionEngine::insertSession(SessionItem *item)
{
    {
        std::lock_guard<std::mutex> g(m_newMu);
        m_newSessions.push_back(item);
    }
    m_cond_has_work.notify_one();
}

void ExecutionEngine::deleteSession(SessionItem *item)
{
    {
        std::lock_guard<std::mutex> g(m_delMu);
        m_deletedSessions.insert(item);
    }
    m_cond_has_work.notify_one();
}

std::future<void> ExecutionEngine::InserterImpl::enqueueOperation(std::unique_ptr<OperationTask> &&task)
{
    std::packaged_task<void()> package(std::bind(&OperationTask::run, task.get()));

    auto opItem = new OperationItem {std::move(task), std::move(package)};
    auto ok = m_item->queue.push(opItem);
    assert(ok);

    m_engine->m_cond_has_work.notify_one();

    return opItem->task.get_future();
}

ExecutionEngine::InserterImpl::~InserterImpl()
{
    if (m_engine && m_item)
    m_engine->deleteSession(m_item);
}

bool ExecutionEngine::shouldWaitForAWhile(bool scheduled)
{
    static auto last = steady_clock::now();

    auto now = steady_clock::now();

    if (scheduled) {
        last = now;
    }

    std::chrono::nanoseconds idle = now - last;
    if (idle > 20ms) {
        INFO("Yielding out due to no progress for {}ms.",
             std::chrono::duration_cast<std::chrono::milliseconds>(idle).count());
        return true;
    }
    return false;
}

void ExecutionEngine::scheduleLoop()
{
    ResourceMonitor resMon;
    resMon.initializeLimits();

    SessionList sessions;


    while (!m_shouldExit) {
        // Fisrt check if there's any pending deletions
        SessionSet del;
        {
            utils::Guard g(m_delMu);
            using std::swap;
            swap(del, m_deletedSessions);
        }

        // Append any new sessions
        {
            utils::Guard g(m_newMu);
            sessions.splice(sessions.end(), m_newSessions);
        }

        // Loop through sessions
        bool scheduled = false;
        size_t count = 0;
        auto it = sessions.begin();
        auto end = sessions.end();
        while (it != end) {
            auto item = *it;
            if (del.count(item)) {
                resMon.clear(item->sessHandle);
                delete item;
                it = sessions.erase(it);
            } else {
                {
                    // Move from lock free queue to backing storage
                    // Use an extra block level to limit the life time
                    // of opItem, so we don't accidentially access it
                    // after the loop.
                    OperationItem *opItem;
                    while (item->queue.pop(opItem)) {
                        item->bgQueue.push(opItem);
                    }
                }
                count += item->bgQueue.size();

                // NOTE: don't use scheduled || maybeScheduleFrom(...)
                // we don't want short-cut eval here and maybeScheduleFrom
                // should always be called.
                scheduled |= maybeScheduleFrom(resMon, item);
                ++it;
            }
        }

        if (shouldWaitForAWhile(scheduled)) {
            // no progress for a long time.
            // gie out our time slice to avoid using too much cycles
//             std::this_thread::yield();
            std::this_thread::sleep_for(10ms);
        }

        if (!count) {
            INFO("Wait on m_cond_has_work");
            std::unique_lock<std::mutex> ul(m_condMu);
            m_cond_has_work.wait(ul);
        }
    }

    // Cleanup
    for (auto item : sessions) {
        while (!item->bgQueue.empty()) {
            delete item->bgQueue.front();
            item->bgQueue.pop();
        }
        item->queue.consume_all([](auto t){
            delete t;
        });
        delete item;
    }
}

bool tryScheduleOn(ResourceMonitor &resMon, OperationTask *t, const std::string &sessHandle,
                   const DeviceSpec &dev)
{
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

    if (expectedDev != dev) {
        // the task wants to run on a different device
        usage = t->estimatedUsage(expectedDev);
        if (!resMon.tryAllocate(usage, sessHandle)) {
            logScheduleFailure(usage, resMon);
            return false;
        }
        if (t->prepare(expectedDev)) {
            return true;
        }
        resMon.free(usage);
    }
    return false;
}

size_t ExecutionEngine::maybeScheduleFrom(ResourceMonitor &resMon, ExecutionEngine::SessionItem* item)
{
    auto &queue = item->bgQueue;

    if (queue.empty()) {
        return 0;
    }

    auto opItem = queue.front();

    // Try schedule the operation
    bool scheduled = false;
    if (useGPU()) {
        if (tryScheduleOn(resMon, opItem->op.get(), item->sessHandle, DeviceType::GPU)) {
            INFO("Task scheduled on GPU");
            scheduled = true;
        }
    }

    if (!scheduled && tryScheduleOn(resMon, opItem->op.get(), item->sessHandle, DeviceType::CPU)) {
        INFO("Task scheduled on CPU");
        scheduled = true;
    }

    // Send to thread pool
    if (scheduled) {
        queue.pop();
        q::with(m_qec->queue(), opItem).then([](OperationItem *opItem){
            opItem->task();
            delete opItem;
        });
    }

    return scheduled ? 1 : 0;
}
