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

#include "resources.h"

#include "platform/logging.h"
#include "utils/containerutils.h"
#include "utils/threadutils.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <tuple>
#include <optional>

using std::optional;
using salus::Guard;

std::string enumToString(const ResourceType &rt)
{
    switch (rt) {
    case ResourceType::COMPUTE:
        return "COMPUTE";
    case ResourceType::MEMORY:
        return "MEMORY";
    default:
        return "Unknown ResourceType";
    }
}

ResourceType resourceTypeFromString(const std::string &rt)
{
    static std::unordered_map<std::string, ResourceType> lookup{
        {"COMPUTE", ResourceType::COMPUTE},
        {"MEMORY", ResourceType::MEMORY},
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
        return {resourceTypeFromString(str), {DeviceType::CPU, 0}};
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
    oss << enumToString(type) << "@" << device.DebugString();
    return oss.str();
}

std::string ResourceMap::DebugString() const
{
    std::ostringstream oss;
    oss << "ResourceMap" << std::endl;
    oss << "    Temporary" << std::endl;
    oss << resources::DebugString(temporary, "        ");
    oss << "    Persistant (handle='" << persistantHandle << "')" << std::endl;
    oss << resources::DebugString(persistant, "        ");
    return oss.str();
}

std::string ResourceMonitor::DebugString() const
{
    std::ostringstream oss;
    oss << "ResourceMonitor: dumping available resources" << std::endl;

    Guard g(m_mu);

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
// Return true iff avail contains req
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

Resources &scale(Resources &lhs, double scale)
{
    for (auto &p : lhs) {
        p.second *= scale;
    }
    return lhs;
}

Resources &removeZeros(Resources &lhs)
{
    auto it = lhs.begin();
    auto itend = lhs.end();
    while (it != itend) {
        if (it->second == 0) {
            it = lhs.erase(it);
        } else {
            ++it;
        }
    }
    return lhs;
}

size_t totalMemory(Resources &res)
{
    size_t mem = 0;
    for (auto p : res) {
        mem += p.second;
    }
    return mem;
}

std::string DebugString(const Resources &res, const std::string &indent)
{
    std::ostringstream oss;
    for (auto p : res) {
        oss << indent << p.first.DebugString() << " -> " << p.second << std::endl;
    }
    return oss.str();
}

} // namespace resources

using namespace resources;

/* static */ SessionResourceTracker &SessionResourceTracker::instance()
{
    static SessionResourceTracker srt;

    return srt;
}

// Read limits from hardware, and capped by cap
SessionResourceTracker::SessionResourceTracker()
{
    // 100 G for CPU
    m_limits[{ResourceType::MEMORY, DeviceType::CPU}] = 100_sz * 1024 * 1024 * 1024;

    // 14 G for GPU 0
    m_limits[{ResourceType::MEMORY, DeviceType::GPU}] = 14_sz * 1024 * 1024 * 1024;
}

SessionResourceTracker::SessionResourceTracker(const Resources &cap)
    : SessionResourceTracker()
{
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

void SessionResourceTracker::setDisabled(bool val)
{
    Guard g(m_mu);
    m_disabled = val;
}

bool SessionResourceTracker::disabled() const
{
    Guard g(m_mu);
    return m_disabled;
}

// If it is safe to admit this session, given its persistant and temporary memory usage.
bool SessionResourceTracker::canAdmitUnsafe(const ResourceMap &cap) const
{
    if (!contains(m_limits, cap.persistant)) {
        return false;
    }

    auto temp(m_limits);
    subtract(temp, cap.persistant);

    if (m_peak.empty()) {
        return true;
    } else {
        return contains(temp, m_peak.front()->temporary);
    }
}

// Take the session
bool SessionResourceTracker::admit(const ResourceMap &cap, uint64_t &ticket)
{
    Guard g(m_mu);

    if (m_disabled) {
        return true;
    }

    if (!canAdmitUnsafe(cap)) {
        return false;
    }

    ticket = ++m_tickets;

    subtract(m_limits, cap.persistant);

    m_sessions[ticket] = cap;

    auto it = m_peak.begin();
    auto itend = m_peak.end();
    while (it != itend && contains((*it)->temporary, cap.temporary)) {
        it++;
    }
    m_peak.insert(it, &m_sessions[ticket]);

    return true;
}

void SessionResourceTracker::acceptAdmission(uint64_t ticket, const std::string &sessHandle)
{
    Guard g(m_mu);
    if (m_disabled) {
        return;
    }

    m_sessions[ticket].persistantHandle = sessHandle;
}

salus::optional<ResourceMap> SessionResourceTracker::usage(uint64_t ticket) const
{
    Guard g(m_mu);

    auto it = m_sessions.find(ticket);
    if (it == m_sessions.end()) {
        return {};
    }

    return it->second;
}

void SessionResourceTracker::freeUnsafe(uint64_t ticket)
{
    auto it = m_sessions.find(ticket);
    if (it == m_sessions.end()) {
        LOG(ERROR) << "SessionResourceTracker: unknown ticket: " << ticket;
        return;
    }

    merge(m_limits, it->second.persistant);

    m_peak.erase(std::remove_if(m_peak.begin(), m_peak.end(),
                 [&it](auto pr) { return pr == &(it->second); }),
                 m_peak.end());

    m_sessions.erase(it);
}

void SessionResourceTracker::free(uint64_t ticket)
{
    AllocLog(INFO) << "Free session resource: ticket=" << ticket;
    Guard g(m_mu);
    if (m_disabled) {
        return;
    }

    freeUnsafe(ticket);
}

std::string SessionResourceTracker::DebugString() const
{
    Guard g(m_mu);

    std::ostringstream oss;

    oss << "SessionResourceTracker" << std::endl;
    oss << "    Issued tickets:" << std::endl;
    for (auto &p : m_sessions) {
        oss << "      " << p.first << " -> " << p.second.DebugString();
    }
    return oss.str();
}

void ResourceMonitor::initializeLimits()
{
    Guard g(m_mu);

    // 100 G for CPU
    m_limits[{ResourceType::MEMORY, DeviceType::CPU}] = 100_sz * 1024 * 1024 * 1024;

    // 14 G for GPU 0
    m_limits[{ResourceType::MEMORY, DeviceType::GPU}] = 14_sz * 1024 * 1024 * 1024;
}

void ResourceMonitor::initializeLimits(const Resources &cap)
{
    initializeLimits();

    Guard g(m_mu);

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

bool ResourceMonitor::preAllocate(const Resources &cap, uint64_t *ticket)
{
    Guard g(m_mu);
    if (!contains(m_limits, cap)) {
        return false;
    }

    *ticket = ++m_nextTicket;

    // Allocate
    subtract(m_limits, cap);
    m_staging[*ticket] = cap;

    return true;
}

bool ResourceMonitor::allocate(uint64_t ticket, const Resources &res)
{
    Guard g(m_mu);
    return allocateUnsafe(ticket, res);
}

bool ResourceMonitor::LockedProxy::allocate(uint64_t ticket, const Resources &res)
{
    assert(m_resMonitor);
    return m_resMonitor->allocateUnsafe(ticket, res);
}

bool ResourceMonitor::allocateUnsafe(uint64_t ticket, const Resources &res)
{
    if (ticket == 0) {
        LOG(ERROR) << "Invalid ticket 0";
        return false;
    }

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
    } else {
        LOG(ERROR) << "Unknown ticket: " << ticket;
    }

    VLOG(2) << "Try allocating from global avail for ticket: " << ticket;

    removeZeros(remaining);

    // ... then try from global avail
    if (!contains(m_limits, remaining)) {
        return false;
    }

    if (it != m_staging.end()) {
        // actual subtract from staging
        auto fromStaging(res);
        subtract(fromStaging, remaining);
        removeZeros(fromStaging);
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
void ResourceMonitor::free(uint64_t ticket)
{
    if (ticket == 0) {
        LOG(ERROR) << "Invalid ticket 0";
        return;
    }

    Guard g(m_mu);

    auto it = m_staging.find(ticket);
    if (it == m_staging.end()) {
        LOG(ERROR) << "Unknown ticket: " << ticket;
        return;
    }

    merge(m_limits, it->second);
    m_staging.erase(it);
}

bool ResourceMonitor::free(uint64_t ticket, const Resources &res)
{
    Guard g(m_mu);
    return freeUnsafe(ticket, res);
}

bool ResourceMonitor::LockedProxy::free(uint64_t ticket, const Resources &res)
{
    assert(m_resMonitor);
    return m_resMonitor->freeUnsafe(ticket, res);
}

bool ResourceMonitor::freeUnsafe(uint64_t ticket, const Resources &res)
{
    // Ticket can not be 0 when free actual resource to prevent
    // monitor go out of sync of physical usage.
    assert(ticket != 0);

    merge(m_limits, res);

    auto it = m_using.find(ticket);
    assert(it != m_using.end());

    assert(contains(it->second, res));

    subtract(it->second, res);
    removeZeros(it->second);
    if (it->second.empty()) {
        m_using.erase(it);
        return true;
    }
    return false;
}

std::vector<std::pair<size_t, uint64_t>> ResourceMonitor::sortVictim(
    const std::unordered_set<uint64_t> &candidates) const
{
    assert(!candidates.empty());

    std::vector<std::pair<size_t, uint64_t>> usages;
    usages.reserve(candidates.size());

    // TODO: currently only select based on GPU memory usage, generalize to all resources
    ResourceTag tag{ResourceType::MEMORY, {DeviceType::GPU, 0}};
    {
        Guard g(m_mu);
        for (auto &ticket : candidates) {
            auto usagemap = salus::optionalGet(m_using, ticket);
            if (!usagemap) {
                continue;
            }
            auto gpuusage = salus::optionalGet(usagemap, tag);
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
    Guard g(m_mu);
    Resources res;
    for (auto t : tickets) {
        merge(res, salus::getOrDefault(m_using, t, {}));
    }
    return res;
}

salus::optional<Resources> ResourceMonitor::queryUsage(uint64_t ticket) const
{
    Guard g(m_mu);
    return salus::optionalGet(m_using, ticket);
}

bool ResourceMonitor::hasUsage(uint64_t ticket) const
{
    Guard g(m_mu);
    return m_using.count(ticket) > 0;
}
