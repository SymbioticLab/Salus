//
// Created by peifeng on 4/17/18.
//

#include "execution/engine/iterationcontext.h"

#include "execution/engine/resourcecontext.h"
#include "execution/scheduler/operationitem.h"
#include "execution/operationtask.h"
#include "platform/logging.h"

namespace salus {

void IterationContext::scheduleTask(std::unique_ptr<OperationTask> &&task)
{
    auto opItem = std::make_shared<OperationItem>();
    opItem->sess = m_item;
    opItem->op = std::move(task);
    LogOpTracing() << "OpItem Event " << opItem->op << " event: queued";

    m_taskExec.queueTask(std::move(opItem));
}

void IterationContext::finish()
{
    if (m_done) {
        m_done(*m_item);
    }
    m_item->endIteration(m_graphId);
}

} // namespace salus
