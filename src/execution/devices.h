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

#ifndef SALUS_EXEC_DEVICES_H
#define SALUS_EXEC_DEVICES_H

#include <string>
#include <tuple>
#include <ostream>

#include "utils/macros.h"

namespace salus {

// NOTE: this order is used in salus::oplib::tensorflow::TFInstance::DeviceContainer
// If modify this, also update there.
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

    DeviceSpec() noexcept = default;

    explicit constexpr DeviceSpec(DeviceType t, int id = 0) noexcept : type(t), id(id) {}

    static DeviceSpec fromString(const std::string &str);

    std::string debugString() const;

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

inline std::ostream &operator<<(std::ostream &os, const DeviceSpec &c)
{
    return os << enumToString(c.type) << ":" << c.id;
}

namespace devices {
constexpr DeviceSpec CPU0 {DeviceType::CPU, 0};
constexpr DeviceSpec GPU0 {DeviceType::GPU, 0};
constexpr DeviceSpec GPU1 {DeviceType::GPU, 1};
} // namespace devices

} // namespace salus

namespace std {
template<>
class hash<salus::DeviceSpec>
{
public:
    inline size_t operator()(const salus::DeviceSpec &spec) const
    {
        size_t res = 0;
        sstl::hash_combine(res, spec.type);
        sstl::hash_combine(res, spec.id);
        return res;
    }
};
} // namespace std


#endif // SALUS_EXEC_DEVICES_H
