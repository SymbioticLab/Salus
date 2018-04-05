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

#ifndef SALUS_EXEC_OPEARTIONTASK_H
#define SALUS_EXEC_OPEARTIONTASK_H

#include "execution/devices.h"
#include "execution/resources.h"

#include <boost/range/any_range.hpp>

#include <functional>
#include <memory>
#include <vector>

class ResourceContext;

namespace salus {
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

    // Estimate usage and cache the result
    virtual Resources estimatedUsage(const DeviceSpec &dev) = 0;

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
