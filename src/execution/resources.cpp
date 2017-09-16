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

#include "utils/threadutils.h"
#include "utils/containerutils.h"
#include "platform/logging.h"

#include <sstream>
#include <tuple>
#include <algorithm>
#include <functional>

using utils::Guard;
using boost::optional;

std::string enumToString(const ResourceType &rt)
{
    switch(rt) {
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

    ResourceTag tag;
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
    for (auto p : temporary) {
        oss << "        " << p.first.DebugString() << " -> " << p.second << std::endl;
    }
    oss << "    Persistant (handle='" << persistantHandle << "')" << std::endl;
    for (auto p : persistant) {
        oss << "        " << p.first.DebugString() << " -> " << p.second << std::endl;
    }
    return oss.str();
}

std::string ResourceMonitor::DebugString() const
{
    std::ostringstream oss;
    oss << "ResourceMonitor: dumping available resources" << std::endl;

    Guard g(m_mu);

    oss << "    Available" << std::endl;
    for (auto p : m_limits) {
        oss << "        ";
        oss << p.first.DebugString() << " -> " << p.second << std::endl;
    }
    oss << "    Staging" << std::endl;
    for (auto p : m_staging) {
        oss << "        ";
        oss << "Ticket: " <<  p.first << std::endl;

        for (auto pp : p.second) {
            oss << "            ";
            oss << pp.first.DebugString() << " -> " << pp.second << std::endl;
        }
    }
    oss << "    In use" << std::endl;
    for (auto p : m_using) {
        oss << "        ";
        oss << "Ticket: " <<  p.first << std::endl;

        for (auto pp : p.second) {
            oss << "            ";
            oss << pp.first.DebugString() << " -> " << pp.second << std::endl;
        }
    }
    return oss.str();
}

namespace resources {
// Return true iff avail contains req
bool contains(const Resources &avail, const Resources &req)
{
    auto aend = avail.end();

    ResourceTag tag;
    double val;
    for (auto p : req) {
        std::tie(tag, val) = p;
        auto it = avail.find(tag);
        if (it == aend && val != 0) {
            return false;
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
    while(it != itend) {
        if (it->second == 0) {
            it = lhs.erase(it);
        } else {
            ++it;
        }
    }
    return lhs;
}

} // namespace resources;

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
    m_limits[{ResourceType::MEMORY, DeviceType::CPU}] = 100.0 * 1024 * 1024 * 1024;

    // 14 G for GPU 0
    m_limits[{ResourceType::MEMORY, DeviceType::GPU}] = 14.0 * 1024 * 1024 * 1024;
}

SessionResourceTracker::SessionResourceTracker(const Resources &cap)
    : SessionResourceTracker()
{
    auto lend = m_limits.end();

    ResourceTag tag;
    double val;
    for (auto p : cap) {
        std::tie(tag, val) = p;
        auto it = m_limits.find(tag);
        if (it != lend) {
            it->second = std::min(it->second, val);
        }
    }
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
    m_sessToTicket[sessHandle] = ticket;
    m_sessions[ticket].persistantHandle = sessHandle;
}

utils::optional<ResourceMap> SessionResourceTracker::usage(const std::string &sessHandle) const
{
    utils::optional<ResourceMap> res;
    auto it = m_sessToTicket.find(sessHandle);
    if (it == m_sessToTicket.end()) {
        return res;
    }

    auto it2 = m_sessions.find(it->second);
    if (it2 == m_sessions.end()) {
        return res;
    }

    res = it2->second;
    return res;
}

void SessionResourceTracker::freeUnsafe(uint64_t ticket)
{
    auto it = m_sessions.find(ticket);
    if (it == m_sessions.end()) {
        ERR("SessionResourceTracker: unknown ticket: {}", ticket);
        return;
    }

    merge(m_limits, it->second.persistant);

    using namespace std::placeholders;

    m_peak.erase(std::remove_if(m_peak.begin(), m_peak.end(), [&it](auto pr) {
        return pr == &(it->second);
    }));

    m_sessions.erase(it);
}

void SessionResourceTracker::free(uint64_t ticket)
{
    DEBUG("Free session resource: ticket={}", ticket);
    Guard g(m_mu);
    freeUnsafe(ticket);
}

void SessionResourceTracker::free(const std::string &sessHandle)
{
    DEBUG("Free session resource: session={}", sessHandle);
    Guard g(m_mu);

    auto it = m_sessToTicket.find(sessHandle);
    if (it == m_sessToTicket.end()) {
        ERR("SessionResourceTracker: unknown sess handle: {}", sessHandle);
        return;
    }

    freeUnsafe(it->second);
}

void ResourceMonitor::initializeLimits()
{
    Guard g(m_mu);

    // 100 G for CPU
    m_limits[{ResourceType::MEMORY, DeviceType::CPU}] = 100.0 * 1024 * 1024 * 1024;

    // 14 G for GPU 0
    m_limits[{ResourceType::MEMORY, DeviceType::GPU}] = 14.0 * 1024 * 1024 * 1024;
}

void ResourceMonitor::initializeLimits(const Resources &cap)
{
    initializeLimits();

    Guard g(m_mu);

    auto lend = m_limits.end();

    ResourceTag tag;
    double val;
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
    if (ticket == 0) {
        ERR("Invalid ticket 0");
        return false;
    }

    auto remaining(res);
    Guard g(m_mu);
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
        ERR("Unknown ticket: {}", ticket);
    }

    WARN("Try allocating from global avail for ticket: {}", ticket);

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
        ERR("Invalid ticket 0");
        return;
    }

    Guard g(m_mu);

    auto it = m_staging.find(ticket);
    if (it == m_staging.end()) {
        ERR("Unknown ticket: {}", ticket);
        return;
    }

    merge(m_limits, it->second);
    m_staging.erase(it);
}

bool ResourceMonitor::free(uint64_t ticket, const Resources &res)
{
    // Ticket can not be 0 when free actual resource to prevent
    // monitor go out of sync of physical usage.
    assert(ticket != 0);

    Guard g(m_mu);
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

std::vector<std::pair<double, uint64_t>> ResourceMonitor::sortVictim(const std::unordered_set<uint64_t> &candidates)
{
    assert(!candidates.empty());

    std::vector<std::pair<double, uint64_t>> usages;
    usages.reserve(candidates.size());

    // TODO: currently only select based on GPU memory usage, generalize to all resources
    ResourceTag tag{ ResourceType::MEMORY, {DeviceType::GPU, 0}};
    {
        Guard g(m_mu);
        for (auto &ticket : candidates) {
            auto usage = utils::getOrDefault(m_using, ticket, utils::optional<Resources>{});
            if (!usage) {
                continue;
            }
            usages.emplace_back(utils::getOrDefault(usage, tag, 0.0), ticket);
        }
    }

    std::sort(usages.begin(), usages.end());
    return usages;
}

utils::optional<Resources> ResourceMonitor::queryUsage(uint64_t ticket)
{
    Guard g(m_mu);
    return utils::getOrDefault(m_using, ticket, utils::optional<Resources>{});
}
