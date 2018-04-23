//
// Created by peifeng on 4/17/18.
//

#ifndef SALUS_EXEC_ITERATIONCONTEXT_H
#define SALUS_EXEC_ITERATIONCONTEXT_H

#include "execution/scheduler/sessionitem.h"

namespace salus {
class OperationTask;
class IterationContext
{
    TaskExecutor &m_taskExec;
    PSessionItem m_item;
    uint64_t m_graphId;

public:
    IterationContext(TaskExecutor &engine, PSessionItem item)
        : m_taskExec(engine)
        , m_item(std::move(item))
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
