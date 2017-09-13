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
 * 
 */

#ifndef DEVICES_H
#define DEVICES_H

#include <string>
#include <tuple>

#include "utils/macros.h"

enum class DeviceType
{
    CPU,
    GPU
};

std::string enumToString(const DeviceType &dt);
DeviceType deviceTypeFromString(const std::string &dt);

struct DeviceSpec
{
    DeviceType type;
    int id;

    DeviceSpec() = default;

    DeviceSpec(DeviceType t, int id = 0) : type(t), id(id) {}

    static DeviceSpec fromString(const std::string &str);

    std::string DebugString() const;

private:
    friend bool operator==(const DeviceSpec &lhs, const DeviceSpec &rhs);
    friend bool operator!=(const DeviceSpec &lhs, const DeviceSpec &rhs);

    auto tie() const { return std::tie(type, id); }
};

inline bool operator==(const DeviceSpec &lhs, const DeviceSpec &rhs)
{
    return lhs.tie() == rhs.tie();
}

inline bool operator!=(const DeviceSpec &lhs, const DeviceSpec &rhs)
{
    return lhs.tie() != rhs.tie();
}

namespace std {
template<>
class hash<DeviceSpec>
{
public:
    inline size_t operator()(const DeviceSpec &spec) const
    {
        size_t res = 0;
        utils::hash_combine(res, spec.type);
        utils::hash_combine(res, spec.id);
        return res;
    }
};
} // namespace std


#endif // DEVICES_H
