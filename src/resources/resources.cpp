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

#include "resources/resources.h"

#include "platform/logging.h"
#include "utils/containerutils.h"
#include "utils/threadutils.h"
#include "utils/debugging.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <tuple>
#include <optional>

using std::optional;
using std::chrono::duration_cast;
using FpMS = std::chrono::duration<double, std::chrono::milliseconds::period>;
using namespace std::chrono_literals;
using namespace date;
using namespace salus;

std::string enumToString(const ResourceType &rt)
{
    switch (rt) {
    case ResourceType::COMPUTE:
        return "COMPUTE";
    case ResourceType::MEMORY:
        return "MEMORY";
    case ResourceType::GPU_STREAM:
        return "GPU_STREAM";
    case ResourceType::EXCLUSIVE:
        return "EXCLUSIVE";
    default:
        return "Unknown ResourceType";
    }
}

ResourceType resourceTypeFromString(const std::string &rt)
{
    static std::unordered_map<std::string, ResourceType> lookup{
        {"COMPUTE", ResourceType::COMPUTE},
        {"MEMORY", ResourceType::MEMORY},
        {"GPU_STREAM", ResourceType::GPU_STREAM},
    };

    auto it = lookup.find(rt);
    if (it != lookup.end()) {
        return it->second;
    }
    return ResourceType::UNKNOWN;
}

/*static*/ ResourceTag ResourceTag::fromString(const std::string &str)
{
    auto pos = str.find(':');
    if (pos == std::string::npos) {
        return {resourceTypeFromString(str), DeviceSpec{DeviceType::CPU, 0}};
    }

    ResourceTag tag{};
    tag.type = resourceTypeFromString(str.substr(0, pos));

    pos = pos + 1;
    if (pos < str.size()) {
        tag.device = DeviceSpec::fromString(str.substr(pos));
    }

    return tag;
}

std::string ResourceTag::DebugString() const
{
    std::ostringstream oss;
    oss << enumToString(type) << "@" << device;
    return oss.str();
}

std::string ResStats::DebugString() const
{
    std::ostringstream oss;
    oss << "ResStats(temporary=" << temporary << ", persist=" << persist << ", count=" << count << ")";
    return oss.str();
}

std::string ResourceMonitor::DebugString() const
{
    std::ostringstream oss;
    oss << "ResourceMonitor: dumping available resources" << std::endl;

    auto g = sstl::with_guard(m_mu);

    oss << "    Available:" << std::endl;
    oss << resources::DebugString(m_limits, "        ");

    oss << "    Staging " << m_staging.size() << " tickets, in total:" << std::endl;
    Resources total;
    for (auto p : m_staging) {
        resources::merge(total, p.second);
    }
    oss << resources::DebugString(total, "       ");

    oss << "    In use " << m_using.size() << " tickets, in total:" << std::endl;
    total.clear();
    for (auto p : m_using) {
        resources::merge(total, p.second);
    }
    oss << resources::DebugString(total, "       ");

    return oss.str();
}

namespace resources {
bool contains(const Resources &avail, const Resources &req)
{
    auto aend = avail.end();

    ResourceTag tag{};
    size_t val;
    for (auto p : req) {
        std::tie(tag, val) = p;
        auto it = avail.find(tag);
        if (it == aend && val != 0) {
            return false;
        }
        if (it == aend) {
            continue;
        }
        if (val > it->second) {
            return false;
        }
    }
    return true;
}

bool compatible(const Resources &lhs, const Resources &rhs)
{
    for (auto p : rhs) {
        if (lhs.count(p.first) == 0) {
            return false;
        }
    }
    return true;
}

Resources &merge(Resources &lhs, const Resources &rhs, bool skipNonExist)
{
    for (auto p : rhs) {
        if (lhs.count(p.first) == 0 && skipNonExist) {
            continue;
        }
        lhs[p.first] += p.second;
    }
    return lhs;
}

Resources &subtract(Resources &lhs, const Resources &rhs, bool skipNonExist)
{
    for (auto p : rhs) {
        if (lhs.count(p.first) == 0 && skipNonExist) {
            continue;
        }
        lhs[p.first] -= p.second;
    }
    return lhs;
}

Resources subtractBounded(Resources &lhs, const Resources &rhs)
{
    const auto lend = lhs.end();
    Resources res;
    for (auto [tag, val] : rhs) {
        auto it = lhs.find(tag);
        if (it == lend) {
            continue;
        }
        auto v = std::min(val, it->second);
        it->second -= v;
        res[tag] = v;
    }
    return res;
}

Resources &scale(Resources &lhs, double scale)
{
    for (auto &p : lhs) {
        p.second *= scale;
    }
    return lhs;
}

Resources &removeInvalid(Resources &lhs)
{
    auto it = lhs.begin();
    auto itend = lhs.end();
    while (it != itend) {
        if (it->second <= 0) {
            it = lhs.erase(it);
        } else {
            ++it;
        }
    }
    return lhs;
}

std::string DebugString(const Resources &res, const std::string &indent)
{
    std::ostringstream oss;
    for (auto p : res) {
        oss << indent << p.first.DebugString() << " -> " << p.second << std::endl;
    }
    return oss.str();
}

Resources platformLimits()
{
    Resources res;
    res[{ResourceType::MEMORY, devices::CPU0}] = 50_sz * 1024 * 1024 * 1024;

    // 15 G for GPU 0
    res[{ResourceType::MEMORY, devices::GPU0}] = 15_sz * 1024 * 1024 * 1024;

    // 80 streams for GPU 0
    res[{ResourceType::GPU_STREAM, devices::GPU0}] = 80;

    res[{ResourceType::EXCLUSIVE, devices::GPU0}] = 1;

    return res;
}

} // namespace resources

using namespace resources;

// Read limits from hardware, and capped by cap
AllocationRegulator::AllocationRegulator()
{
    m_limits = resources::platformLimits();
}

AllocationRegulator::AllocationRegulator(const Resources &cap)
    : AllocationRegulator()
{
    auto lend = m_limits.end();

    for (auto [tag, val] : cap) {
        auto it = m_limits.find(tag);
        if (it != lend) {
            it->second = std::min(it->second, val);
        }
    }
}

AllocationRegulator::Ticket AllocationRegulator::registerJob()
{
    auto g = sstl::with_guard(m_mu);
    Ticket t{++m_next, this};
    m_jobs.try_emplace(t);
    return t;
}

bool AllocationRegulator::Ticket::beginAllocation(const Resources &res)
{
    {
        auto g = sstl::with_guard(reg->m_mu);

        if (!contains(reg->m_limits, res)) {
            return false;
        }

        subtract(reg->m_limits, res);

        merge(reg->m_jobs[*this].inuse, res);
    }
    LogAlloc() << "Start session allocation hold: ticket=" << as_int
            << ", res=" << sstl::getOrDefault(res, resources::GPU0Memory, 0);

    return true;
}

void AllocationRegulator::Ticket::endAllocation(const Resources &res)
{
    Resources released;
    {
        auto g = sstl::with_guard(reg->m_mu);

        auto it = reg->m_jobs.find(*this);
        // HACK: finishJob may be called earlier than endAllocation
        if (it == reg->m_jobs.end()) {
            return;
        }
        auto &js = it->second;

        released = subtractBounded(js.inuse, res);

        removeInvalid(js.inuse);
        merge(reg->m_limits, released);
    }
    LogAlloc() << "End session allocation hold: ticket=" << as_int
            << ", res=" << sstl::getOrDefault(released, resources::GPU0Memory, 0);
}

void AllocationRegulator::Ticket::finishJob()
{
    auto g = sstl::with_guard(reg->m_mu);

    if (auto it = reg->m_jobs.find(*this); it != reg->m_jobs.end()) {
        merge(reg->m_limits, it->second.inuse);
        reg->m_jobs.erase(it);
    }
}

std::string AllocationRegulator::DebugString() const
{
    auto g = sstl::with_guard(m_mu);

    std::ostringstream oss;

    oss << "AllocationRegulator(Free:" << m_limits << std::endl;
    oss << "    Issued tickets:" << std::endl;
    for (const auto &[ticket, state] : m_jobs) {
        oss << "      " << ticket.as_int << " -> " << state.inuse;
    }
    oss << ")";
    return oss.str();
}

void ResourceMonitor::initializeLimits()
{
    auto g = sstl::with_guard(m_mu);

    m_limits = resources::platformLimits();
}

void ResourceMonitor::initializeLimits(const Resources &cap)
{
    initializeLimits();

    auto g = sstl::with_guard(m_mu);

    auto lend = m_limits.end();

    ResourceTag tag{};
    size_t val;
    for (auto p : cap) {
        std::tie(tag, val) = p;
        auto it = m_limits.find(tag);
        if (it != lend) {
            it->second = std::min(it->second, val);
        }
    }
}

std::optional<uint64_t> ResourceMonitor::preAllocate(const Resources &req, Resources *missing)
{
    // TODO: check ticket

    auto g = sstl::with_guard(m_mu);
    if (!contains(m_limits, req)) {
        if (missing) {
            *missing = req;
            subtract(*missing, m_limits, true /* skipNonExist */);
            removeInvalid(*missing);
        }
        return {};
    }

    auto ticket = ++m_nextTicket;

    // Allocate
    subtract(m_limits, req);
    m_staging[ticket] = req;

    return ticket;
}

bool ResourceMonitor::allocate(uint64_t ticket, const Resources &res)
{
    if (ticket == 0) {
        LOG(ERROR) << "Invalid ticket 0";
        return false;
    }

    auto g = sstl::with_uguard(m_mu);
    return allocateUnsafe(ticket, res);
}

bool ResourceMonitor::LockedProxy::allocate(uint64_t ticket, const Resources &res)
{
    assert(m_resMonitor);
    if (ticket == 0) {
        LOG(ERROR) << "Invalid ticket 0";
        return false;
    }

    return m_resMonitor->allocateUnsafe(ticket, res);
}

bool ResourceMonitor::allocateUnsafe(uint64_t ticket, const Resources &res)
{
    auto remaining(res);
    auto it = m_staging.find(ticket);
    if (it != m_staging.end()) {
        // first try allocate from reserve
        if (contains(it->second, remaining)) {
            subtract(it->second, remaining);
            merge(m_using[ticket], remaining);
            return true;
        }

        // pre-allocation is not enough, see how much we need
        // to request from global avail...
        subtract(remaining, it->second, true /*skipNonExist*/);
    }

    removeInvalid(remaining);

    // ... then try from global avail
    if (!contains(m_limits, remaining)) {
        return false;
    }

    if (it != m_staging.end()) {
        // actual subtract from staging
        auto fromStaging(res);
        subtract(fromStaging, remaining);
        removeInvalid(fromStaging);
        assert(contains(it->second, fromStaging));
        subtract(it->second, fromStaging);
    }

    // actual subtract from global
    subtract(m_limits, remaining);

    // add to used
    merge(m_using[ticket], res);

    return true;
}

// Release remaining pre-allocated resources
void ResourceMonitor::freeStaging(uint64_t ticket)
{
    if (ticket == 0) {
        LOG(ERROR) << "Invalid ticket 0 for freeStaging";
        return;
    }

    auto g = sstl::with_uguard(m_mu);

    auto it = m_staging.find(ticket);
    if (it == m_staging.end()) {
        g.unlock();
        LOG(ERROR) << "Unknown ticket for freeStaging: " << ticket;
        return;
    }

    merge(m_limits, it->second);
    m_staging.erase(it);
}

bool ResourceMonitor::free(uint64_t ticket, const Resources &res)
{
    auto g = sstl::with_guard(m_mu);
    return freeUnsafe(ticket, res);
}

bool ResourceMonitor::LockedProxy::free(uint64_t ticket, const Resources &res)
{
    assert(m_resMonitor);
    return m_resMonitor->freeUnsafe(ticket, res);
}

std::optional<Resources> ResourceMonitor::LockedProxy::queryStaging(uint64_t ticket) const
{
    DCHECK(m_resMonitor);
    return m_resMonitor->queryStagingUnsafe(ticket);
}

bool ResourceMonitor::freeUnsafe(uint64_t ticket, const Resources &res)
{
    // Ticket can not be 0 when free actual resource to prevent
    // monitor go out of sync of physical usage.
    DCHECK_NE(ticket, 0);

    merge(m_limits, res);

    auto it = m_using.find(ticket);
    DCHECK_NE(it, m_using.end());

    DCHECK(contains(it->second, res));

    subtract(it->second, res);
    removeInvalid(it->second);
    if (it->second.empty()) {
        m_using.erase(it);
        return true;
    }
    return false;
}

std::optional<Resources> ResourceMonitor::queryStagingUnsafe(uint64_t ticket) const
{
    DCHECK_NE(ticket, 0);
    return sstl::optionalGet(m_staging, ticket);
}

std::vector<std::pair<size_t, uint64_t>> ResourceMonitor::sortVictim(
    const std::unordered_set<uint64_t> &candidates) const
{
    assert(!candidates.empty());

    std::vector<std::pair<size_t, uint64_t>> usages;
    usages.reserve(candidates.size());

    // TODO: currently only select based on GPU memory usage, generalize to all resources
    ResourceTag tag{ResourceType::MEMORY, devices::GPU0};
    {
        auto g = sstl::with_guard(m_mu);
        for (auto &ticket : candidates) {
            auto usagemap = sstl::optionalGet(m_using, ticket);
            if (!usagemap) {
                continue;
            }
            auto gpuusage = sstl::optionalGet(usagemap, tag);
            if (!gpuusage || *gpuusage == 0) {
                continue;
            }
            usages.emplace_back(*gpuusage, ticket);
        }
    }

    std::sort(usages.begin(), usages.end(), [](const auto &lhs, const auto &rhs) {
        return lhs > rhs;
    });
    return usages;
}

Resources ResourceMonitor::queryUsages(const std::unordered_set<uint64_t> &tickets) const
{
    auto g = sstl::with_guard(m_mu);
    Resources res;
    for (auto t : tickets) {
        merge(res, sstl::getOrDefault(m_using, t, {}));
    }
    return res;
}

optional<Resources> ResourceMonitor::queryUsage(uint64_t ticket) const
{
    auto g = sstl::with_guard(m_mu);
    return sstl::optionalGet(m_using, ticket);
}

bool ResourceMonitor::hasUsage(uint64_t ticket) const
{
    auto g = sstl::with_guard(m_mu);
    return m_using.count(ticket) > 0;
}
