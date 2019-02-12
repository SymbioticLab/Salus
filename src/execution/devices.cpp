/*
 * Copyright 2019 Peifeng Yu <peifeng@umich.edu>
 * 
 * This file is part of Salus
 * (see https://github.com/SymbioticLab/Salus).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

std::string DeviceSpec::debugString() const
{
    return enumToString(type) + ":" + std::to_string(id);
}

} // namespace salus
