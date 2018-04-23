//
// Created by peifeng on 4/18/18.
//

#ifndef SALUS_EXEC_ITERATIONTASK_H
#define SALUS_EXEC_ITERATIONTASK_H

#include "resources/resources.h"
#include "utils/pointerutils.h"

#include <string>

namespace salus {
class IterationContext;
class ExecutionEngine;
class IterationTask
{
public:
    virtual ~IterationTask();

    virtual uint64_t graphId() const = 0;

    virtual bool prepare() = 0;

    virtual ResStats estimatedPeakAllocation(const DeviceSpec &dev) const = 0;

    virtual void runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept = 0;

    virtual void cancel() {};

    virtual bool isCanceled() const = 0;
};

} // namespace salus

#endif // SALUS_EXEC_ITERATIONTASK_H
