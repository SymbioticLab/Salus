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

    virtual bool isExpensive() const = 0;
};

} // namespace salus

#endif // SALUS_EXEC_ITERATIONTASK_H
