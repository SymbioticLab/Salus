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
#include "platform/logging.h"

#include <sstream>
#include <tuple>
#include <algorithm>
#include <algorithm>
#include <functional>

using utils::Guard;

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

ResourceType enumFromString(const std::string &rt)
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

// Return true iff avail contains req
bool contains(const Resources &avail, const Resources &req)
{
    auto aend = avail.end();

    ResourceTag tag;
    double val;
    for (auto p : req) {
        std::tie(tag, val) = p;
        auto it = avail.find(tag);
        if (it == aend) {
            return false;
        }
        if (val > it->second) {
            return false;
        }
    }
    return true;
}

Resources &merge(Resources &lhs, const Resources &rhs)
{
    for (auto p : rhs) {
        lhs[p.first] += p.second;
    }
    return lhs;
}

Resources &subtract(Resources &lhs, const Resources &rhs)
{
    for (auto p : rhs) {
        lhs[p.first] -= p.second;
    }
    return lhs;
}

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

    m_persist[ticket] = cap;

    auto it = m_peak.begin();
    auto itend = m_peak.end();
    while (it != itend && contains((*it)->temporary, cap.temporary)) {
        it++;
    }
    m_peak.insert(it, &m_persist[ticket]);

    return true;
}

void SessionResourceTracker::acceptAdmission(uint64_t ticket, const std::string &sessHandle)
{
    Guard g(m_mu);
    m_sessions[sessHandle] = ticket;
    m_persist[ticket].persistantHandle = sessHandle;
}

void SessionResourceTracker::freeUnsafe(uint64_t ticket)
{
    auto it = m_persist.find(ticket);
    if (it == m_persist.end()) {
        WARN("SessionResourceTracker: unknown ticket: {}", ticket);
        return;
    }

    merge(m_limits, it->second.persistant);

    using namespace std::placeholders;

    m_peak.erase(std::remove_if(m_peak.begin(), m_peak.end(), [&it](auto pr) {
        return pr == &(it->second);
    }));

    m_persist.erase(it);
}

void SessionResourceTracker::free(uint64_t ticket)
{
    Guard g(m_mu);
    freeUnsafe(ticket);
}

void SessionResourceTracker::free(const std::string &sessHandle)
{
    Guard g(m_mu);

    auto it = m_sessions.find(sessHandle);
    if (it == m_sessions.end()) {
        WARN("SessionResourceTracker: unknown sess handle: {}", sessHandle);
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

// Try aquare resources in as specified cap, including persistant resources.
// Persistant resources will be allocated under handle
// return false if failed, no resource will be allocated
bool ResourceMonitor::tryAllocate(const ResourceMap &cap, const std::string &handle)
{
    Guard g(m_mu);

    auto &persist = m_persis[handle];

    auto toAlloc = cap.temporary;

    // Check if persistant resources has been allocated.
    if (persist.count(cap.persistantHandle) == 0) {
        merge(toAlloc, cap.persistant);
    }

    // Check if we have enough resource
    if (!contains(m_limits, toAlloc)) {
        return false;
    }

    // Allocate
    for (auto p : toAlloc) {
        m_limits[p.first] -= p.second;
    }

    // Record persistant
    persist[cap.persistantHandle] = cap.persistant;

    return true;
}

// Free non persistant resources
void ResourceMonitor::free(const ResourceMap &cap)
{
    Guard g(m_mu);

    for (auto p : cap.temporary) {
        m_limits[p.first] += p.second;
    }
}

// Free persistant resources under handle
void ResourceMonitor::clear(const std::string &handle)
{
    Guard g(m_mu);

    auto &persist = m_persis[handle];

    for (auto p : persist) {
        merge(m_limits, p.second);
    }
    persist.clear();
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

    for (auto p : m_limits) {
        oss << "    ";
        oss << p.first.DebugString() << " -> " << p.second << std::endl;
    }
    return oss.str();
}
