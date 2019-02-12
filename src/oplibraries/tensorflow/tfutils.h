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

#ifndef SALUS_OPLIB_TENSORFLOW_TFUTILS_H
#define SALUS_OPLIB_TENSORFLOW_TFUTILS_H

#include "execution/devices.h"

#include <functional>
#include <memory>
#include <string_view>

#define CallWithMasterMethodName(m)                                                                                    \
    m(CreateSession) m(ExtendSession) m(PartialRunSetup) m(CloseSession) m(ListDevices) m(Reset) m(RunStep)

namespace tensorflow {
#define FWD_DECLARE(name)                                                                                              \
    class name##Request;                                                                                               \
    class name##Response;

CallWithMasterMethodName(FWD_DECLARE)

#undef FWD_DECLARE

class Status;
class DeviceType;
class OpKernel;
class Graph;
} // namespace tensorflow

namespace perftools::gputools {
} // namespace perftools::gputools

namespace salus::oplib::tensorflow {

namespace tf = ::tensorflow;
namespace tfgpu = ::perftools::gputools;

using Status = tf::Status;
using StatusCallback = std::function<void(Status)>;

#define DECLARE_USING(name)                                                                                            \
    using P##name##Request = std::unique_ptr<tf::name##Request>;                                                       \
    using P##name##Response = std::unique_ptr<tf::name##Response>;                                                     \
    using name##Callback = std::function<void(P##name##Response &&, Status)>;

CallWithMasterMethodName(DECLARE_USING)

#undef DECLARE_USING

DeviceSpec tfDeviceNameToSpec(const std::string &name);
DeviceType tfDeviceTypeToType(const tf::DeviceType &type);
DeviceType tfDeviceTypeToType(const std::string &type);

using POpKernel = std::unique_ptr<tf::OpKernel, void (*)(tf::OpKernel *)>;

std::string tfGraphToGraphviz(const tf::Graph &g, const std::string &name);

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_TFUTILS_H
