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
    std::string m_name;

public:
    IterationContext(TaskExecutor &engine, PSessionItem item)
        : m_taskExec(engine)
        , m_item(std::move(item))
    {
    }

    ~IterationContext() = default;

    void scheduleTask(std::unique_ptr<OperationTask> &&task);

    void setName(const std::string &name)
    {
        m_name = name;
    }

    const std::string &name() const
    {
        return m_name;
    }

    void finish();
};

} // namespace salus

#endif // SALUS_EXEC_ITERATIONCONTEXT_H
