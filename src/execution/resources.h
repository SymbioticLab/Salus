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
#include <mutex>
#include <list>

enum class ResourceType
{
    COMPUTE,
    MEMORY,

    UNKNOWN = 1000,
};

std::string enumToString(const ResourceType &rt);
ResourceType enumFromString(const std::string &rt);

struct ResourceTag
{
    ResourceType type;
    DeviceSpec device;

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

using Resources = std::unordered_map<ResourceTag, double>;

namespace resources {
// Return true iff avail contains req
bool contains(const Resources &avail, const Resources &req);

Resources &merge(Resources &lhs, const Resources &rhs);
Resources &subtract(Resources &lhs, const Resources &rhs);
Resources &scale(Resources &lhs, double scale);
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

    // Try aquare resources in as specified cap, including persistant resources.
    // Persistant resources will be allocated under handle
    // return false if failed, no resource will be allocated
    bool tryAllocate(const ResourceMap &cap, const std::string &handle);

    // Free non persistant resources
    void free(const ResourceMap &cap);

    // Free persistant resources under handle
    void clear(const std::string &handle);

    std::string DebugString() const;

private:
    mutable std::mutex m_mu;

    Resources m_limits;

    // Map from persistantHandle -> InnerMap
    using PerSessInnerMap = std::unordered_map<std::string, Resources>;

    // Map from session -> PerSessInnerMap
    std::unordered_map<std::string, PerSessInnerMap> m_persis;
};

#endif // EXECUTION_RESOURCES_H
