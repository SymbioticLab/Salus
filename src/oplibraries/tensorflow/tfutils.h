/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Aetf <aetf@unlimitedcodeworks.xyz>
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

#ifndef SALUS_OPLIB_TENSORFLOW_TFUTILS_H
#define SALUS_OPLIB_TENSORFLOW_TFUTILS_H

#include "execution/devices.h"
#include <functional>
#include <memory>
#include <string_view>

#define CallWithMasterMethodName(m)                                                                          \
    m(CreateSession) m(ExtendSession) m(PartialRunSetup) m(CloseSession) m(ListDevices) m(Reset) m(RunStep)

namespace tensorflow {
#define FWD_DECLARE(name)                                                                                    \
    class name##Request;                                                                                     \
    class name##Response;

CallWithMasterMethodName(FWD_DECLARE)

#undef FWD_DECLARE

    class Status;
} // namespace tensorflow

namespace salus::oplib::tensorflow {

namespace tf = ::tensorflow;

using Status = tf::Status;
using StatusCallback = std::function<void(Status)>;

#define DECLARE_USING(name)                                                                                  \
    using P##name##Request = std::unique_ptr<tf::name##Request>;                                             \
    using P##name##Response = std::unique_ptr<tf::name##Response>;                                           \
    using name##Callback = std::function<void(P##name##Response &&, Status)>;

CallWithMasterMethodName(DECLARE_USING)

#undef DECLARE_USING

DeviceSpec tfDeviceNameToSpec(const std::string &name);
DeviceType tfDeviceTypeToType(const tf::DeviceType &type);
DeviceType tfDeviceTypeToType(const std::string &type);

inline tf::StringPiece svToStringPiece(std::string_view sv)
{
    return {sv.data(), sv.size()};
}

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_TFUTILS_H
