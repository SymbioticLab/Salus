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

using std::chrono::steady_clock;
using namespace std::chrono_literals;

namespace {
void logScheduleFailure(const ResourceMap &usage, const ResourceMonitor &resMon)
{
    DEBUG("Try to allocate resource failed. Requested: {}", usage.DebugString());
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
    m_note_has_work.notify();
}

void ExecutionEngine::deleteSession(SessionItem *item)
{
    {
        std::lock_guard<std::mutex> g(m_delMu);
        m_deletedSessions.insert(item);
    }
    m_note_has_work.notify();
}

std::future<void> ExecutionEngine::InserterImpl::enqueueOperation(std::unique_ptr<OperationTask> &&task)
{
    using namespace std::placeholders;
    using DoneCallback = OperationTask::DoneCallback;
    std::packaged_task<void(DoneCallback, DoneCallback)> package(std::bind(&OperationTask::run,
                                                                           task.get(),
                                                                           _1, _2));

    auto opItem = new OperationItem {std::move(task), std::move(package)};
    auto ok = m_item->queue.push(opItem);
    assert(ok);

    m_engine->m_note_has_work.notify();

    return opItem->task.get_future();
}

ExecutionEngine::InserterImpl::~InserterImpl()
{
    if (m_engine && m_item)
    m_engine->deleteSession(m_item);
}

bool ExecutionEngine::shouldWaitForAWhile(bool scheduled, std::chrono::nanoseconds &ns)
{
    static auto last = steady_clock::now();
    static auto sleep = 10ms;

    auto now = steady_clock::now();

    if (scheduled) {
        last = now;
        sleep = 10ms;
    }

    std::chrono::nanoseconds idle = now - last;
    if (idle > 20ms) {
        INFO("Yielding out due to no progress for {}ms.",
             std::chrono::duration_cast<std::chrono::milliseconds>(idle).count());
        ns = sleep;
        sleep *= 2;
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
                    // TODO: if one session adds op too quickly, it may block here.
                    OperationItem *opItem;
                    while (item->queue.pop(opItem)) {
                        item->bgQueue.push_back(opItem);
                    }
                }
                count += item->bgQueue.size();

                // NOTE: don't use scheduled || maybeScheduleFrom(...)
                // we don't want short-cut eval here and maybeScheduleFrom
                // should always be called.
                scheduled |= maybeScheduleFrom(resMon, item) > 0;
                ++it;
            }
        }

        std::chrono::nanoseconds ns;
        if (shouldWaitForAWhile(scheduled, ns)) {
            // no progress for a long time.
            // gie out our time slice to avoid using too much cycles
//             std::this_thread::yield();
            std::this_thread::sleep_for(ns);
        }

        if (!count) {
            INFO("Wait on m_note_has_work");
            m_note_has_work.wait();
        }
    }

    // Cleanup
    for (auto item : sessions) {
        delete item;
    }
}

ExecutionEngine::SessionItem::~SessionItem()
{
    while (!bgQueue.empty()) {
        delete bgQueue.front();
        bgQueue.pop_front();
    }
    queue.consume_all([](auto t){
        delete t;
    });
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
    return false;
}

size_t ExecutionEngine::maybeScheduleFrom(ResourceMonitor &resMon, ExecutionEngine::SessionItem* item)
{
    auto &queue = item->bgQueue;

    auto size = queue.size();

    if (size == 0) {
        return 0;
    }

    // Try schedule the operation
    auto doSchedule = [&resMon, &item, this](OperationItem *opItem) -> OperationItem* {
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
        if (!scheduled) {
            return opItem;
        } else {
            q::with(m_qec->queue(), opItem).then([spec, item, &resMon](OperationItem *opItem){
                opItem->task([&resMon, opItem, spec](){
                    // succeed
                    ResourceMap res;
                    opItem->op->lastUsage(spec, res);
                    resMon.free(res);
                    delete opItem;
                }, [item, opItem](){
                    // failed due to OOM. Push back to queue
                    item->queue.push(opItem);
                    WARN("Opkernel {} failed due to OOM", opItem->op->DebugString());
                });
            });
            return nullptr;
        }
    };

    // Do all schedule in queue in parallel
    UnsafeQueue stage;
    stage.swap(queue);
    std::vector<q::promise<OperationItem*>> promises;
    for (auto opItem : stage) {
        auto p = q::with(m_qec->queue(), opItem).then(doSchedule);
        promises.emplace_back(std::move(p));
    }

    assert(queue.empty());

    auto it = queue.begin();
    utils::notification n;
    q::all(std::move(promises), m_qec->queue())
    .then([it, &n](std::vector<OperationItem*> remain) {
        std::remove_copy(remain.begin(), remain.end(), it, nullptr);
        n.notify();
    });
    n.wait();

    return size - queue.size();
}
