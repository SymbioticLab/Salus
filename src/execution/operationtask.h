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

#ifndef SALUS_EXEC_OPEARTIONTASK_H
#define SALUS_EXEC_OPEARTIONTASK_H

#include "execution/devices.h"
#include "resources/resources.h"

#include <boost/range/any_range.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace salus {
class ResourceContext;
class OperationTask
{
public:
    using DoneCallback = std::function<void(void)>;
    using MemFailCallback = std::function<bool(void)>;
    struct Callbacks
    {
        DoneCallback done;
        MemFailCallback memFailure;
    };

    virtual ~OperationTask();

    virtual std::string DebugString() const = 0;

    virtual uint64_t graphId() const = 0;

    // Estimate usage and cache the result
    virtual Resources estimatedUsage(const DeviceSpec &dev) = 0;
    virtual bool hasExactEstimation(const DeviceSpec &dev) = 0;

    // All supported device types for this task
    using DeviceTypes =
        boost::any_range<DeviceType, boost::forward_traversal_tag, DeviceType &, std::ptrdiff_t>;
    virtual DeviceTypes supportedDeviceTypes() const = 0;

    virtual int failedTimes() const = 0;

    virtual bool prepare(std::unique_ptr<ResourceContext> &&rctx) noexcept = 0;

    virtual ResourceContext &resourceContext() const = 0;

    // If allow paging happen when this task is running.
    virtual bool isAsync() const = 0;

    virtual void run(Callbacks cbs) noexcept = 0;

    virtual void cancel() = 0;
};

inline std::ostream& operator<<(std::ostream& out, const OperationTask& op)
{
    return out << op.DebugString();
}

inline std::ostream& operator<<(std::ostream& out, const std::unique_ptr<OperationTask>& op)
{
    return out << op->DebugString();
}

} // namespace salus

#endif // SALUS_EXEC_OPEARTIONTASK_H
