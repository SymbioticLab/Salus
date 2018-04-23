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

#include "execution/executionengine.h"

#include "execution/engine/iterationcontext.h"
#include "execution/engine/resourcecontext.h"
#include "execution/iterationtask.h"
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
#include <boost/container/flat_map.hpp>

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
        // Engine has been iterrupted
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

void ExecutionEngine::scheduleLoop()
{
    LOG(INFO) << "ExecutionEngine scheduling thread started";
    // a thread local queue
    IterQueue iters;
    // a higher priority queue. Only one session's iter in this queue can run, and will block iters queue
    boost::circular_buffer<std::pair<PSessionItem, boost::circular_buffer<IterationItem>>>
        blockingSessions(128);

    // staging queue
    IterQueue staging;

    while (true) {
        size_t scheduled = 0;
        // accept new iters
        {
            auto g = sstl::with_guard(m_mu);
            staging.swap(m_iterQueue);
        }

        // move things to aproriate queue
        for (auto &iter : staging) {
            auto ectx = iter.wectx.lock();
            if (!ectx) {
                continue;
            }
            if (ectx->m_item->exlusiveMode) {
                bool found = false;
                for (auto &[psess, queue] : blockingSessions) {
                    if (psess == ectx->m_item) {
                        found = true;
                        queue.push_back(std::move(iter));
                        break;
                    }
                }
                if (!found) {
                    blockingSessions.push_back();
                    blockingSessions.back().first = ectx->m_item;
                    blockingSessions.back().second.set_capacity(4);
                    blockingSessions.back().second.push_back(std::move(iter));
                }
            } else {
                iters.emplace_back(std::move(iter));
            }
        }
        staging.clear();

        if (m_interrupting) {
            break;
        }

        // schedule in blocking queue
        bool hasBlocking = false;
        while (!blockingSessions.empty()) {
            auto &[pSess, queue] = blockingSessions.front();
            if (queue.empty() && !pSess->exlusiveMode) {
                // this session has done with exlusive mode
                blockingSessions.pop_front();
                continue;
            }
            hasBlocking = true;

            while (!queue.empty()) {
                auto &iterItem = queue.front();

                if (iterItem.iter->isCanceled()) {
                    queue.pop_front();
                    continue;
                }

                auto ectx = iterItem.wectx.lock();
                if (!ectx) {
                    queue.pop_front();
                    continue;
                }

                if (runIter(iterItem, *ectx)) {
                    queue.pop_front();
                    scheduled += 1;
                }
            }
            break;
        }

        if (hasBlocking && scheduled > 0) {
            maybeWaitForAWhile(scheduled);

            bool noWork = blockingSessions.empty() && iters.empty();
            if (noWork) {
                for (auto &[pSess, queue] : blockingSessions) {
                    UNUSED(pSess);
                    if (!queue.empty()) {
                        noWork = false;
                        break;
                    }
                }
            }
            if (noWork) {
                VLOG(2) << "ExecutionEngine wait on m_note_has_work";
                m_note_has_work.wait();
            }
            continue;
        }

        iters.swap(staging);
        for (auto &iterItem : staging) {
            if (iterItem.iter->isCanceled()) {
                continue;
            }

            auto ectx = iterItem.wectx.lock();
            if (!ectx) {
                continue;
            }

            // automatically place back if we can't schedule to run it by the end of this iteration
            sstl::ScopeGuards put_back([&]() { iters.emplace_back(std::move(iterItem)); });

            if (runIter(iterItem, *ectx)) {
                put_back.dismiss();
                scheduled += 1;
            }
        } // for (auto &[wectx, iter] : staging) {

        maybeWaitForAWhile(scheduled);

        bool noWork = blockingSessions.empty() && iters.empty();
        if (noWork) {
            for (auto &[pSess, queue] : blockingSessions) {
                UNUSED(pSess);
                if (!queue.empty()) {
                    noWork = false;
                    break;
                }
            }
        }
        if (noWork) {
            VLOG(2) << "ExecutionEngine wait on m_note_has_work";
            m_note_has_work.wait();
        }
    }

    // Cleanup
    {
        // make sure no more new iters are pending
        auto g = sstl::with_guard(m_mu);
        iters.splice(iters.end(), m_iterQueue);
    }

    for (const auto &iterItem : iters) {
        iterItem.iter->cancel();
    }
    for (const auto &[pSess, queue] : blockingSessions) {
        UNUSED(pSess);
        for (const auto &iterItem : queue) {
            iterItem.iter->cancel();
        }
    }
    LOG(INFO) << "ExecutionEngine stopped";
}

bool ExecutionEngine::runIter(IterationItem &iterItem, ExecutionContext &ectx)
{
    DCHECK(ectx.m_item);

    // FUTURE: support other devices
    if (!iterItem.iter->prepare()) {
        return false;
    }

    auto iCtx = std::make_shared<IterationContext>(m_taskExecutor, ectx.m_item);
    iterItem.iter->runAsync(std::move(iCtx));
    return true;
}

bool ExecutionEngine::maybeWaitForAWhile(size_t scheduled)
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

std::unique_ptr<ResourceContext> ExecutionContext::makeResourceContext(const std::string &graphId,
                                                                       const DeviceSpec &spec,
                                                                       const Resources &res,
                                                                       Resources *missing)
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
} // namespace salus
