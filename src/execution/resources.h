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

#include <unordered_map>

enum class ResourceType
{
    COMPUTE,
    MEMORY,
};

std::string enumToString(const ResourceType &rt);

struct ResourceTag
{
    ResourceType type;
    DeviceSpec device;
    bool persistant = false;

    std::string DebugString() const;

private:
    friend bool operator==(const ResourceTag &lhs, const ResourceTag &rhs);
    friend bool operator!=(const ResourceTag &lhs, const ResourceTag &rhs);

    auto tie() const { return std::tie(type, device, persistant); }
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
        utils::hash_combine(res, tag.persistant);
        return res;
    }
};
} // namespace std

using ResourceMap = std::unordered_map<ResourceTag, double>;

/**
 * A monitor of resources. This class is not thread-safe.
 */
class ResourceMonitor
{
public:
    ResourceMonitor() = default;

    // Read limits from hardware, and capped by cap
    void initializeLimits();
    void initializeLimits(const ResourceMap &cap);

    // If the resource described by cap is available
    bool available(const ResourceMap &cap);

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
    struct Tag
    {
    private:
        auto tie() const { return std::tie(type, device); }

    public:
        ResourceType type;
        DeviceSpec device;

        static Tag fromRTag(const ResourceTag &rtag)
        {
            return {rtag.type, rtag.device};
        }

        std::string DebugString() const;

        bool operator==(const Tag &rhs) const { return tie() == rhs.tie(); }
        bool operator!=(const Tag &rhs) const { return tie() != rhs.tie(); }
    };

    struct hasher
    {
    public:
        inline size_t operator()(const Tag &tag) const
        {
            size_t res = 0;
            utils::hash_combine(res, tag.type);
            utils::hash_combine(res, tag.device);
            return res;
        }
    };

    using InnerMap = std::unordered_map<Tag, double, hasher>;

    // Return true iff avail contains req
    bool contains(const InnerMap &avail, const ResourceMap &req);

    InnerMap m_limits;
    std::unordered_map<std::string, InnerMap> m_persis;
};

#endif // EXECUTION_RESOURCES_H
