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
        return "ResourceType::COMPUTE";
    case ResourceType::MEMORY:
        return "ResourceType::COMPUTE";
    default:
        return "Unknown ResourceType";
    }
}

// Return true iff avail contains req
bool ResourceMonitor::contains(const InnerMap &avail, const ResourceMap &req)
{
    auto aend = avail.end();

    ResourceTag rtag;
    double val;
    for (auto p : req) {
        std::tie(rtag, val) = p;
        auto it = avail.find(Tag::fromRTag(rtag));
        if (it == aend) {
            return false;
        }
        if (val > it->second) {
            return false;
        }
    }
    return true;
}

void ResourceMonitor::initializeLimits()
{
    // 100 G for CPU
    m_limits[{ResourceType::MEMORY, DeviceType::CPU}] = 100.0 * 1024 * 1024 * 1024;

    // 14 G for GPU 0
    m_limits[{ResourceType::MEMORY, DeviceType::GPU}] = 14.0 * 1024 * 1024 * 1024;
}

void ResourceMonitor::initializeLimits(const ResourceMap &cap)
{
    initializeLimits();

    auto lend = m_limits.end();

    ResourceTag rtag;
    double val;
    for (auto p : cap) {
        std::tie(rtag, val) = p;
        auto it = m_limits.find(Tag::fromRTag(rtag));
        if (it != lend) {
            it->second = std::min(it->second, val);
        }
    }
}

// If the resource described by cap is available
bool ResourceMonitor::available(const ResourceMap &cap)
{
    return contains(m_limits, cap);
}

// Try aquare resources in as specified cap, including persistant resources.
// Persistant resources will be allocated under handle
// return false if failed, no resource will be allocated
bool ResourceMonitor::tryAllocate(const ResourceMap &cap, const std::string &handle)
{
    if (!available(cap)) {
        return false;
    }

    ResourceTag rtag;
    double val;
    for (auto p : cap) {
        std::tie(rtag, val) = p;

        auto tag = Tag::fromRTag(rtag);
        m_limits[tag] -= val;
        if (rtag.persistant) {
            m_persis[handle][tag] += val;
        }
    }
    return true;
}

// Free non persistant resources
void ResourceMonitor::free(const ResourceMap &cap)
{
    ResourceTag rtag;
    double val;
    for (auto p : cap) {
        std::tie(rtag, val) = p;
        if (!rtag.persistant) {
            auto tag = Tag::fromRTag(rtag);
            m_limits[tag] += val;
        }
    }
}

// Free persistant resources under handle
void ResourceMonitor::clear(const std::string &handle)
{
    auto it = m_persis.find(handle);
    if (it == m_persis.end()) {
        return;
    }

    for (auto p : it->second) {
        m_limits[p.first] += p.second;
    }

    m_persis.erase(it);
}

std::string ResourceMonitor::Tag::DebugString() const
{
    std::ostringstream oss;
    oss << enumToString(type) << "@" << device.DebugString();
    return oss.str();
}

std::string ResourceTag::DebugString() const
{
    std::ostringstream oss;
    oss << enumToString(type) << "@" << device.DebugString() << "(persistant=" << persistant << ")";
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
