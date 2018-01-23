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

#ifndef SESSIONITEM_H
#define SESSIONITEM_H

#include "utils/containerutils.h"
#include "execution/resources.h"
#include "execution/executionengine.h"

#include <list>
#include <string>
#include <functional>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <any>

/**
 * @todo write docs
 */
struct SessionItem
{
    using KernelQueue = std::list<POpItem>;
    using UnsafeQueue = std::list<POpItem>;
private:
    // protected by mu (may be accessed both in schedule thread and close session thread)
    PagingCallbacks pagingCb;
    std::function<void()> cleanupCb;
    KernelQueue queue;
    std::mutex mu;

    size_t lastScheduled = 0;

    uint64_t holWaiting = 0;
    uint64_t queueHeadHash = 0;

    std::unordered_set<uint64_t> tickets;
    std::mutex tickets_mu;

    // Accessed by multiple scheduling thread
    std::atomic_bool protectOOM{true};

    friend class ExecutionEngine;
    friend class ResourceContext;
    friend class IScheduler;

public:
    std::string sessHandle;

    // Only accessed by main scheduling thread
    UnsafeQueue bgQueue;
    bool forceEvicted{false};

    explicit SessionItem(const std::string &handle)
        : sessHandle(handle)
    {
        // NOTE: add other devices
        resUsage[resources::GPU0Memory].get() = 0;
        resUsage[resources::CPU0Memory].get() = 0;
    }

    ~SessionItem();

    utils::MutableAtom::value_type &resourceUsage(const ResourceTag &tag)
    {
        return resUsage.at(tag).get();
    }

private:
    using AtomicResUsages = std::unordered_map<ResourceTag, utils::MutableAtom>;
    // must be initialized in constructor
    AtomicResUsages resUsage;
};
using PSessionItem = std::shared_ptr<SessionItem>;
using SessionList = std::list<PSessionItem>;
using SessionSet = std::unordered_set<PSessionItem>;

#endif // SESSIONITEM_H
