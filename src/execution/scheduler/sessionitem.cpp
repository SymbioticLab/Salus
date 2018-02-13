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
        sstl::Guard g(mu);
        cb = std::move(cleanupCb);
    }
    if (cb) {
        cb();
    }
}

void SessionItem::setPagingCallbacks(PagingCallbacks pcb)
{
    sstl::Guard g(mu);
    pagingCb = std::move(pcb);
}

void SessionItem::prepareDelete(std::function<void()> cb)
{
    sstl::Guard g(mu);
    cleanupCb = std::move(cb);
    // clear paging callbacks so the executorImpl won't get called after it is deleted
    // but haven't been removed from session list yet.
    pagingCb = {};
}
