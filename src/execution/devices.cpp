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

#include "devices.h"

#include "platform/logging.h"
#include "utils/cpp17.h"

#include <sstream>
#include <unordered_map>

namespace salus {

std::string enumToString(const DeviceType &dt)
{
    switch (dt) {
    case DeviceType::CPU:
        return "CPU";
    case DeviceType::GPU:
        return "GPU";
    default:
        return "Unknown DeviceType";
    }
}

DeviceType deviceTypeFromString(const std::string &rt)
{
    static std::unordered_map<std::string, DeviceType> lookup{
        {"CPU", DeviceType::CPU},
        {"GPU", DeviceType::GPU},
    };

    auto it = lookup.find(rt);
    if (it != lookup.end()) {
        return it->second;
    }
    // TODO: add an unknown device type
    return DeviceType::CPU;
}

/*static*/ DeviceSpec DeviceSpec::fromString(const std::string &str)
{
    auto pos = str.find(':');
    if (pos == std::string::npos) {
        return DeviceSpec{deviceTypeFromString(str), 0};
    }

    DeviceSpec spec{deviceTypeFromString(str.substr(0, pos))};

    auto fcr = sstl::from_chars(str.c_str(), str.c_str() + str.size(), spec.id);
    if (fcr.ec) {
        LOG(ERROR) << "Failed to convert '" << str << "' to DeviceSpec";
    }

    return spec;
}

std::string DeviceSpec::DebugString() const
{
    std::ostringstream oss;
    oss << enumToString(type) << ":" << id;
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const DeviceSpec &c)
{
    return os << enumToString(c.type) << ":" << c.id;
}

} // namespace salus
