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

#include <sstream>
#include <tuple>
#include <algorithm>

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

void ResourceMonitor::initializeLimits()
{
    // 100 G for CPU
    m_limits[{ResourceType::MEMORY, DeviceType::CPU}] = 100.0 * 1024 * 1024 * 1024;

    // 14 G for GPU 0
    m_limits[{ResourceType::MEMORY, DeviceType::GPU}] = 14.0 * 1024 * 1024 * 1024;
}

void ResourceMonitor::initializeLimits(const Resources &cap)
{
    initializeLimits();

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

// Return true iff avail contains req
bool ResourceMonitor::contains(const Resources &avail, const Resources &req) const
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

Resources &ResourceMonitor::merge(Resources &lhs, const Resources &rhs) const
{
    for (auto p : rhs) {
        lhs[p.first] += p.second;
    }
    return lhs;
}

// Try aquare resources in as specified cap, including persistant resources.
// Persistant resources will be allocated under handle
// return false if failed, no resource will be allocated
bool ResourceMonitor::tryAllocate(const ResourceMap &cap, const std::string &handle)
{
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
    for (auto p : cap.temporary) {
        m_limits[p.first] += p.second;
    }
}

// Free persistant resources under handle
void ResourceMonitor::clear(const std::string &handle)
{
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

    for (auto p : m_limits) {
        oss << "    ";
        oss << p.first.DebugString() << " -> " << p.second << std::endl;
    }
    return oss.str();
}
