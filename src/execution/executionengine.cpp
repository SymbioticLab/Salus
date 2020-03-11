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

#include "oplibraries/tensorflow/device/gpu/lane/lanemgr.h"

#include "execution/executionengine.h"

#include "execution/engine/iterationcontext.h"
#include "execution/engine/resourcecontext.h"
#include "execution/iterationtask.h"
#include "platform/logging.h"
#include "platform/thread_annotations.h"
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
#include <unordered_map>

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

ExecutionEngine &ExecutionEngine::instance()
{
    static ExecutionEngine eng;
    return eng;
}

ExecutionEngine::ExecutionEngine()
    : m_taskExecutor(m_pool, m_resMonitor, m_schedParam)
{
}

void ExecutionEngine::startScheduler()
{
    m_resMonitor.initializeLimits();
    m_taskExecutor.startExecution();

    m_schedThread = std::make_unique<std::thread>(std::bind(&ExecutionEngine::scheduleLoop, this));
}

void ExecutionEngine::stopScheduler()
{
    m_interrupting = true;

    // unblock scheduling thread
    m_note_has_work.notify();

    if (m_schedThread->joinable()) {
        m_schedThread->join();
    }

    m_taskExecutor.stopExecution();
}

ExecutionEngine::~ExecutionEngine()
{
    stopScheduler();
}

std::shared_ptr<ExecutionContext> ExecutionEngine::makeContext()
{
    if (m_interrupting) {
        return nullptr;
    }

    auto ticket = m_allocReg.registerJob();
    return std::make_shared<ExecutionContext>(*this, ticket);
}

void ExecutionEngine::scheduleIteration(IterationItem &&item)
{
    if (m_interrupting) {
        item.iter->cancel();
        return;
    }

    {
        auto g = sstl::with_guard(m_mu);
        m_iterQueue.emplace_back(std::move(item));
    }
    m_note_has_work.notify();
}

void ExecutionEngine::maybeWaitForWork(size_t pending, size_t scheduled)
{
    maybeWaitForAWhile(scheduled);

    if (pending == 0) {
        VLOG(2) << "ExecutionEngine wait on m_note_has_work";
        m_note_has_work.wait();
    }
}

void ExecutionEngine::scheduleLoop()
{
    LOG(INFO) << "ExecutionEngine scheduling thread started";
    threading::set_thread_name("ExecutionEngine");

    // a map of lane id to thread local queues.
    std::unordered_map<uint64_t, LaneQueue> queues;
    queues.reserve(15);

    // staging queue
    IterQueue staging;

    while (true) {
        DCHECK(staging.empty());
        // accept new iters
        {
            auto g = sstl::with_guard(m_mu);
            staging.swap(m_iterQueue);
        }

        // record the timestamp
        auto currStamp = system_clock::now();

        // move things to aproriate queue
        for (auto &iter : staging) {
            auto ectx = iter.wectx.lock();
            if (!ectx) {
                continue;
            }
            auto &lane = queues[ectx->laneId()];
            lane.queue.emplace_back(std::move(iter));
            lane.lastSeen = currStamp;
            if (lane.sessions.emplace(ectx->m_item).second) {
                lane.id = ectx->laneId();
                // new session to lane and remove old one
                auto it = lane.sessions.begin();
                auto ed = lane.sessions.end();
                while (it != ed) {
                    if (auto s = it->lock()) {
                        s->numFinishedIters = 0;
                        ++it;
                    } else {
                        it = lane.sessions.erase(it);
                    }
                }
                // put session to the back of fifo
                lane.fifoQueue.emplace_back(ectx->m_item);
            }
        }
        staging.clear();

        // break if interrupting, after accepting every thing
        if (m_interrupting) {
            break;
        }

        // schedule each lane separately, release lane that is inactive for too long.
        // it doesn't matter if later that lane has iter comes, just recreate it.

        size_t scheduled = 0;
        size_t pending = 0;

        constexpr const auto MaxInactiveTime = 10s;
        for (auto it = queues.begin(); it != queues.end();) {
            auto &lctx = it->second;
            if (lctx.queue.empty()
                && currStamp - lctx.lastSeen > MaxInactiveTime
                && lctx.numExpensiveIterRunning.load(std::memory_order_acquire) == 0) {
                it = queues.erase(it);
            } else {
                scheduled += scheduleOnQueue(lctx, staging);
                staging.clear();
                pending += lctx.queue.size();
                ++it;
            }
        }

        maybeWaitForWork(pending, scheduled);
    }

    // Cleanup
    {
        // make sure no more new iters are pending
        auto g = sstl::with_guard(m_mu);
        staging.splice(staging.end(), m_iterQueue);
    }

    for (const auto &iterItem : staging) {
        iterItem.iter->cancel();
    }
    LOG(INFO) << "ExecutionEngine stopped";
}

int ExecutionEngine::scheduleOnQueue(LaneQueue &lctx, IterQueue &staging)
{
    lctx.queue.swap(staging);
    CHECK(lctx.queue.empty());

    int scheduled = 0;

    // First let go every mainIter=false
    for (auto &iterItem : staging) {
        if (iterItem.iter->isCanceled()) {
            continue;
        }
        auto ectx = iterItem.wectx.lock();
        if (!ectx) {
            continue;
        }

        if (!iterItem.iter->isExpensive()) {
            if (runIter(iterItem, *ectx, lctx)) {
                scheduled += 1;
                continue;
            }
        }
        // put back all mainIter=true
        lctx.queue.emplace_back(std::move(iterItem));
    }
    staging.clear();

    // For all main iters
    lctx.queue.swap(staging);

    auto rrSorter = [](const auto &iterItemA, const auto &iterItemB) {
        std::shared_ptr<ExecutionContext> ectxA = iterItemA.wectx.lock();
        std::shared_ptr<ExecutionContext> ectxB = iterItemB.wectx.lock();
        if (!ectxB) {
            return true; // A goes first
        }
        if (!ectxA) {
            return false; // B goes first
        }

        return ectxA->m_item->numFinishedIters < ectxB->m_item->numFinishedIters;
    };

    auto fairSorter = [](const auto &iterItemA, const auto &iterItemB) {
        std::shared_ptr<ExecutionContext> ectxA = iterItemA.wectx.lock();
        std::shared_ptr<ExecutionContext> ectxB = iterItemB.wectx.lock();
        if (!ectxB) {
            return true; // A goes first
        }
        if (!ectxA) {
            return false; // B goes first
        }

        // fairness (equalize time)
        auto reA = ectxA->m_item->usedRunningTime.load();
        auto reB = ectxB->m_item->usedRunningTime.load();
        return reA < reB;
    };

    if (m_schedParam.scheduler == "fair") {
        staging.sort(fairSorter);
    } else if (m_schedParam.scheduler == "rr") {
        staging.sort(rrSorter);
    } else if (m_schedParam.scheduler == "pack") {
        // do nothing
    } else if (m_schedParam.scheduler == "fifo") {
        PSessionItem sessItem = nullptr;
        auto it = lctx.fifoQueue.begin();
        auto ed = lctx.fifoQueue.end();
        while (it != ed && !(sessItem = it->lock())) {
            it = lctx.fifoQueue.erase(it);
        }
        if (sessItem) {
            if (lctx.lastSessionItem != sessItem.get()) {
                lctx.lastSessionItem = sessItem.get();
                LOG(INFO) << "event: fifo_select_sess "
                          << as_json({
                                                {"sess", sessItem->sessHandle},
                                                {"laneId", lctx.id},
                                            });
            }
            for (auto &iterItem : staging) {
                if (iterItem.iter->isCanceled()) {
                    continue;
                }
                auto ectx = iterItem.wectx.lock();
                if (!ectx) {
                    continue;
                }
                if (ectx->m_item != sessItem) {
                    lctx.queue.emplace_back(std::move(iterItem));
                    continue;
                }

                if (runIter(iterItem, *ectx, lctx)) {
                    scheduled += 1;
                } else {
                    lctx.queue.emplace_back(std::move(iterItem));
                }
            }
        }
        staging.clear();
    } else {
        CHECK_EQ(m_schedParam.scheduler, "preempt") << "Unknown scheduler selected: " << m_schedParam.scheduler;
        // find the sessItem with least remaining time
        int64_t minRemainingTime = std::numeric_limits<int64_t>::max();
        PSessionItem sessItem = nullptr;
        for (auto &ws : lctx.sessions) {
            if (auto s = ws.lock()) {
                auto remain = static_cast<int64_t>(s->totalRunningTime) - static_cast<int64_t>(s->usedRunningTime);
                if (remain <= minRemainingTime) {
                    minRemainingTime = remain;
                    sessItem = std::move(s);
                }
            }
        }
        if (sessItem) {
            if (lctx.lastSessionItem != sessItem.get()) {
                lctx.lastSessionItem = sessItem.get();
                LOG(INFO) << "event: preempt_select_sess "
                          << as_json({
                                                {"sess", sessItem->sessHandle},
                                                {"totalRunningTime", sessItem->totalRunningTime},
                                                {"usedRunningTime", sessItem->usedRunningTime.load()},
                                                {"laneId", lctx.id},
                                            });
            }
            for (auto &iterItem : staging) {
                if (iterItem.iter->isCanceled()) {
                    continue;
                }
                auto ectx = iterItem.wectx.lock();
                if (!ectx) {
                    continue;
                }
                if (ectx->m_item != sessItem) {
                    lctx.queue.emplace_back(std::move(iterItem));
                    continue;
                }

                if (runIter(iterItem, *ectx, lctx)) {
                    scheduled += 1;
                } else {
                    lctx.queue.emplace_back(std::move(iterItem));
                }
            }
        }
        staging.clear();
    }

    // If work conservation is disabled we will only schedule one iter
    bool done = false;
    for (auto &iterItem : staging) {
        if (iterItem.iter->isCanceled()) {
            continue;
        }

        auto ectx = iterItem.wectx.lock();
        if (!ectx) {
            continue;
        }

        if (!m_schedParam.workConservative && done) {
            lctx.queue.emplace_back(std::move(iterItem));
            continue;
        }

        if (runIter(iterItem, *ectx, lctx)) {
            scheduled += 1;
            if (!m_schedParam.workConservative) {
                done = true;
            }
        } else {
            lctx.queue.emplace_back(std::move(iterItem));
        }
    } // for (auto &iterItem : staging)
    staging.clear();

    return scheduled;
}

bool ExecutionEngine::checkIter(IterationItem &iterItem, ExecutionContext &, LaneQueue &lctx)
{
    if (!iterItem.iter->isExpensive()) {
        return true;
    }

    // only allow one expensive iteration to run
    int64_t zero = 0;
    return lctx.numExpensiveIterRunning.compare_exchange_weak(zero, 1);
}

bool ExecutionEngine::runIter(IterationItem &iterItem, ExecutionContext &ectx, LaneQueue &lctx)
{
    DCHECK(ectx.m_item);

    VLOG(2) << "Try iteration " << ectx.m_item->sessHandle << ":" << iterItem.iter->graphId();
    if (!checkIter(iterItem, ectx, lctx)) {
        VLOG(2) << "event: skip_iter "
                << as_json({{"sess", ectx.m_item->sessHandle},
                                   {"graphId", iterItem.iter->graphId()},
                                   {"reason", "unavailable resources"}});
        return false;
    }

    // FUTURE: support other devices
    if (!iterItem.iter->prepare()) {
        VLOG(2) << "event: skip_iter "
                << as_json({{"sess", ectx.m_item->sessHandle},
                                   {"graphId", iterItem.iter->graphId()},
                                   {"reason", "failed prepare"}});
        return false;
    }

    bool expensive = iterItem.iter->isExpensive();

    auto iCtx = std::make_shared<IterationContext>(m_taskExecutor, ectx.m_item,
                                                   [&lctx, expensive, start = system_clock::now()](auto &sessItem) {
                                                       if (expensive) {
                                                           auto usedTime =
                                                               duration_cast<milliseconds>(system_clock::now() - start).count();
                                                           sessItem.usedRunningTime += usedTime;
                                                           ++sessItem.numFinishedIters;
                                                           if (VLOG_IS_ON(1)) {
                                                               LogOpTracing() << "event: sess_add_time " << as_json({
                                                                   {"sess", sessItem.sessHandle},
                                                                   {"usedRunningTime", sessItem.usedRunningTime.load()},
                                                                   {"totalRunningTime", sessItem.totalRunningTime},
                                                               });
                                                           }
                                                           // NOTE: lctx must be alive when any iters on it finishes.
                                                           lctx.numExpensiveIterRunning--;
                                                       }
                                                   });
    iterItem.iter->runAsync(std::move(iCtx));
    return true;
}

bool ExecutionEngine::maybeWaitForAWhile(size_t scheduled)
{
    static constexpr auto initialSleep = 10ms;
    static constexpr auto getBored = 1s;

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

    if (sleep > 1s) {
        LOG(WARNING) << "No progress for " << duration_cast<milliseconds>(idle).count() << "ms, sleep for "
                     << duration_cast<milliseconds>(sleep).count() << "ms";
    }

    // no progress for a long time.
    // give out our time slice to avoid using too much cycles
    //             std::this_thread::yield();
    std::this_thread::sleep_for(sleep);

    return true;
}

ExecutionContext::ExecutionContext(ExecutionEngine &engine, AllocationRegulator::Ticket ticket)
    : m_engine(engine)
    , m_ticket(ticket)
    , m_item(std::make_shared<SessionItem>(""))
{
}

void ExecutionContext::registerPagingCallbacks(PagingCallbacks &&pcb)
{
    DCHECK(m_item);
    m_item->setPagingCallbacks(std::move(pcb));
}

void ExecutionContext::setInterruptCallback(std::function<void()> cb)
{
    DCHECK(m_item);
    m_item->setInterruptCallback(std::move(cb));
}

std::unique_ptr<ResourceContext> ExecutionContext::makeResourceContext(uint64_t graphId, const DeviceSpec &spec,
                                                                       const Resources &res, Resources *missing)
{
    DCHECK(m_item);
    return m_engine.m_taskExecutor.makeResourceContext(m_item, graphId, spec, res, missing);
}

void ExecutionContext::finish(std::function<void()> cb)
{
    DCHECK(m_item);
    m_item->prepareDelete(std::move(cb));
    // Request taskExec to remove session and give up our reference to the session item
    removeFromEngine();
}

void ExecutionContext::removeFromEngine()
{
    if (m_item) {
        m_engine.m_taskExecutor.deleteSession(std::move(m_item));
    }
    if (m_ticket) {
        m_ticket.finishJob();
        m_ticket = 0;
    }
}

void ExecutionContext::setSessionHandle(const std::string &h)
{
    DCHECK(m_item);
    m_item->sessHandle = h;

    LogAlloc() << "Session " << h << " has tracker ticket " << m_ticket.as_int;

    m_engine.m_taskExecutor.insertSession(m_item);
}

void ExecutionContext::scheduleIteartion(std::unique_ptr<IterationTask> &&iterTask)
{
    m_engine.scheduleIteration({shared_from_this(), std::move(iterTask)});
}

void ExecutionContext::dropExlusiveMode()
{
    DCHECK(m_item);
    m_item->setExclusiveMode(false);
    m_engine.m_note_has_work.notify();
}

void ExecutionContext::setExpectedRunningTime(uint64_t time)
{
    DCHECK(m_item);
    m_item->totalRunningTime = time;
}
} // namespace salus
