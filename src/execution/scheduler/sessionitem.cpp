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
#include "utils/pointerutils.h"

SessionItem::~SessionItem()
{
    bgQueue.clear();
    queue.clear();

    std::function<void()> cb;
    {
        auto g = sstl::with_guard(mu);
        cb = std::move(cleanupCb);
    }
    if (cb) {
        cb();
        // reset cb to release anything that may depend on this
        // before going out of destructor.
        cb = nullptr;
    }

    // output stats
    VLOG(2) << "Stats for Session " << sessHandle << ": totalExecutedOp=" << totalExecutedOp;
}

void SessionItem::setPagingCallbacks(PagingCallbacks pcb)
{
    auto g = sstl::with_guard(mu);
    pagingCb = std::move(pcb);
}

void SessionItem::prepareDelete(std::function<void()> cb)
{
    auto g = sstl::with_guard(mu);
    cleanupCb = std::move(cb);
    // clear paging callbacks so the executorImpl won't get called after it is deleted
    // but haven't been removed from session list yet.
    pagingCb = {};
}

void SessionItem::notifyMemoryAllocation(uint64_t ticket)
{
    auto g = sstl::with_guard(tickets_mu);
    tickets.emplace(ticket);
}

void SessionItem::removeMemoryAllocationTicket(uint64_t ticket)
{
    VLOG(2) << "Removing ticket " << ticket << " from session " << sessHandle;
    auto g = sstl::with_guard(tickets_mu);
    tickets.erase(ticket);
}
