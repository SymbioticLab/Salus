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

#include "sessionitem.h"

#include "platform/logging.h"

using namespace salus;

SessionItem::~SessionItem()
{
    bgQueue.clear();
    queue.clear();

    // output stats
    VLOG(2) << "Stats for Session " << sessHandle << ": totalExecutedOp=" << totalExecutedOp;
}

void SessionItem::setPagingCallbacks(PagingCallbacks pcb)
{
    auto g = sstl::with_guard(mu);
    pagingCb = std::move(pcb);
}

void SessionItem::setInterruptCallback(std::function<void()> cb)
{
    auto g = sstl::with_guard(mu);
    interruptCb = std::move(cb);
}

void SessionItem::prepareDelete(std::function<void()> cb)
{
    setExclusiveMode(false);
    auto g = sstl::with_guard(mu);
    cleanupCb = std::move(cb);
    // clear paging callbacks so the executorImpl won't get called after it is deleted
    // but haven't been removed from session list yet.
    pagingCb = {};
}

void SessionItem::interrupt()
{
    if (forceEvicted) {
        return;
    }
    forceEvicted = true;

    std::function<void()> cb;
    {
        auto g = sstl::with_guard(mu);
        cb = std::move(interruptCb);
    }
    if (cb) cb();
}

void SessionItem::queueTask(POpItem &&opItem)
{
    auto g = sstl::with_guard(mu);
    queue.emplace_back(std::move(opItem));
}

void SessionItem::notifyAlloc(const uint64_t graphId, uint64_t ticket, const ResourceTag &tag, size_t num)
{
    resourceUsage(tag) += num;

    {
        auto g = sstl::with_guard(tickets_mu);
        tickets.emplace(ticket);
    }

    updateTracker(graphId, tag);
}

void SessionItem::notifyDealloc(const uint64_t graphId, uint64_t ticket, const ResourceTag &tag, size_t num, bool last)
{
    resourceUsage(tag) -= num;
    if (last) {
        VLOG(2) << "Removing ticket " << ticket << " from session " << sessHandle;
        auto g = sstl::with_guard(tickets_mu);
        tickets.erase(ticket);
    }

    updateTracker(graphId, tag);
}

void SessionItem::updateTracker(const uint64_t graphId, const ResourceTag &tag)
{
    if (tag == trackerTag) {
        VLOG(2) << "SessionItem::updateTracker graphid=" << graphId << ", sess=" << sessHandle;
        auto g = sstl::with_guard(mu);
        auto it = allocTrackers.find(graphId);
        if (it != allocTrackers.end()) {
            it->second.update(resourceUsage(tag));
        }
    }
}

bool SessionItem::beginIteration(AllocationRegulator::Ticket t, ResStats newRm, const uint64_t graphId)
{
    VLOG(2) << "SessionItem::beginIteration graphid=" << graphId << ", sess=" << sessHandle;
    auto g = sstl::with_guard(mu);
    auto it = allocTrackers.try_emplace(graphId, trackerTag).first;
    return it->second.beginIter(t, newRm, resourceUsage(trackerTag));
}

void SessionItem::endIteration(const uint64_t graphId)
{
    VLOG(2) << "SessionItem::endIteration graphid=" << graphId << ", sess=" << sessHandle;
    auto g = sstl::with_guard(mu);
    allocTrackers.at(graphId).endIter();
}
