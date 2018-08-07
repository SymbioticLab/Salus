//
// Created by peifeng on 4/17/18.
//

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

    using DoneCallback = std::function<void ()>;
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
