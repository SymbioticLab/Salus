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
 */

#include "sessionitem.h"

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
        VLOG(2) << "SessionItem::updateTracker graphid=" << graphId;
        auto g = sstl::with_guard(mu);
        auto it = allocTrackers.find(graphId);
        if (it != allocTrackers.end()) {
            it->second.update(resourceUsage(tag));
        }
    }
}

bool SessionItem::beginIteration(AllocationRegulator::Ticket t, ResStats newRm, const uint64_t graphId)
{
    VLOG(2) << "SessionItem::beginIteration graphid=" << graphId;
    auto g = sstl::with_guard(mu);
    auto it = allocTrackers.try_emplace(graphId, trackerTag).first;
    return it->second.beginIter(t, newRm);
}

void SessionItem::endIteration(const uint64_t graphId)
{
    VLOG(2) << "SessionItem::endIteration graphid=" << graphId;
    auto g = sstl::with_guard(mu);
    allocTrackers.at(graphId).endIter();
}
