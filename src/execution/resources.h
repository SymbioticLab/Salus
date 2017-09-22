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

#ifndef EXECUTION_RESOURCES_H
#define EXECUTION_RESOURCES_H

#include "execution/devices.h"
#include "utils/macros.h"
#include "utils/cpp17.h"

#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <list>
#include <vector>

enum class ResourceType
{
    COMPUTE,
    MEMORY,

    UNKNOWN = 1000,
};

std::string enumToString(const ResourceType &rt);
ResourceType resourceTypeFromString(const std::string &rt);

struct ResourceTag
{
    ResourceType type;
    DeviceSpec device;

    static ResourceTag fromString(const std::string &str);

    std::string DebugString() const;

private:
    friend bool operator==(const ResourceTag &lhs, const ResourceTag &rhs);
    friend bool operator!=(const ResourceTag &lhs, const ResourceTag &rhs);

    auto tie() const { return std::tie(type, device); }
};

inline bool operator==(const ResourceTag &lhs, const ResourceTag &rhs)
{
    return lhs.tie() == rhs.tie();
}

inline bool operator!=(const ResourceTag &lhs, const ResourceTag &rhs)
{
    return lhs.tie() != rhs.tie();
}

namespace std {
template<>
class hash<ResourceTag>
{
public:
    inline size_t operator()(const ResourceTag &tag) const
    {
        size_t res = 0;
        utils::hash_combine(res, tag.type);
        utils::hash_combine(res, tag.device);
        return res;
    }
};
} // namespace std

using Resources = std::unordered_map<ResourceTag, size_t>;

namespace resources {
// Return true iff avail contains req
bool contains(const Resources &avail, const Resources &req);

// Return true iff lhs's tags is superset of rhs's tags
bool compatible(const Resources &lhs, const Resources &rhs);

// Remove items whose value is 0
Resources &removeZeros(Resources &lhs);

Resources &merge(Resources &lhs, const Resources &rhs, bool skipNonExist=false);
Resources &subtract(Resources &lhs, const Resources &rhs, bool skipNonExist=false);
Resources &scale(Resources &lhs, double scale);

std::string DebugString(const Resources &res, const std::string &indent = "");
} // namespace resources

struct ResourceMap
{
    Resources temporary;
    Resources persistant;
    std::string persistantHandle;

    std::string DebugString() const;
};

class SessionResourceTracker
{
    SessionResourceTracker();
    // Read limits from hardware, and capped by cap
    explicit SessionResourceTracker(const Resources &cap);

    // If it is safe to admit this session, given its persistant and temporary memory usage.
    bool canAdmitUnsafe(const ResourceMap &cap) const;

    void freeUnsafe(uint64_t ticket);

public:
    static SessionResourceTracker &instance();

    ~SessionResourceTracker() = default;

    // Take the session
    bool admit(const ResourceMap &cap, uint64_t &ticket);

    // Associate ticket with handle
    void acceptAdmission(uint64_t ticket, const std::string &sessHandle);

    // Query the usage of session.
    utils::optional<ResourceMap> usage(const std::string &sessHandle) const;

    // Free the session
    void free(const std::string &sessHandle);
    void free(uint64_t ticket);

    std::string DebugString() const;

private:
    mutable std::mutex m_mu;

    uint64_t m_tickets = 0;

    Resources m_limits;

    std::unordered_map<std::string, uint64_t> m_sessToTicket;
    std::unordered_map<uint64_t, ResourceMap> m_sessions;

    std::list<ResourceMap*> m_peak;
};

/**
 * A monitor of resources. This class is thread-safe.
 */
class ResourceMonitor
{
public:
    ResourceMonitor() = default;

    // Read limits from hardware, and capped by cap
    void initializeLimits();
    void initializeLimits(const Resources &cap);

    // Try pre-allocate resources
    bool preAllocate(const Resources &cap, uint64_t *ticket);

    // Allocate resources from pre-allocated resources, if res < reserved, gauranteed to succeed
    // otherwise may return false
    bool allocate(uint64_t ticket, const Resources &res);

    // Release remaining pre-allocated resources
    void free(uint64_t ticket);

    // Free resources, return true if after this, the ticket hold no more resources.
    bool free(uint64_t ticket, const Resources &res);

    std::vector<std::pair<size_t, uint64_t>> sortVictim(const std::unordered_set<uint64_t> &candidates) const;

    utils::optional<Resources> queryUsage(uint64_t ticket) const;
    bool hasUsage(uint64_t ticket) const;

    std::string DebugString() const;

private:
    mutable std::mutex m_mu;

    // 0 is invalid ticket
    uint64_t m_nextTicket = 1;
    Resources m_limits;

    std::unordered_map<uint64_t, Resources> m_staging;

    std::unordered_map<uint64_t, Resources> m_using;
};

#endif // EXECUTION_RESOURCES_H
