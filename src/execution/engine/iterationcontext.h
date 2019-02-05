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

#ifndef SALUS_EXEC_ITERATIONCONTEXT_H
#define SALUS_EXEC_ITERATIONCONTEXT_H

#include "execution/scheduler/sessionitem.h"

#include <functional>

namespace salus {
class OperationTask;
class IterationContext
{
    TaskExecutor &m_taskExec;
    PSessionItem m_item;
    uint64_t m_graphId;

    using DoneCallback = std::function<void (SessionItem &)>;
    DoneCallback m_done;

public:
    IterationContext(TaskExecutor &taskExec, PSessionItem item, DoneCallback done)
        : m_taskExec(taskExec)
        , m_item(std::move(item))
        , m_done(std::move(done))
    {
    }

    ~IterationContext() = default;

    void scheduleTask(std::unique_ptr<OperationTask> &&task);

    void setGraphId(uint64_t graphId)
    {
        m_graphId = graphId;
    }

    uint64_t graphId() const
    {
        return m_graphId;
    }

    void finish();
};

} // namespace salus

#endif // SALUS_EXEC_ITERATIONCONTEXT_H
