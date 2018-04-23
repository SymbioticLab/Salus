//
// Created by peifeng on 4/16/18.
//

#include "oplibraries/tensorflow/v2/itertask.h"
#include "execution/engine/iterationcontext.h"

#include <sstream>

namespace salus::oplib::tensorflow {

IterTask::IterTask(ExecutorImpl &impl, const tf::Executor::Args &args, tf::Executor::DoneCallback done)
    : m_impl(impl)
    , m_cm(*args.cancellation_manager)
    , m_args(args)
    , m_done(std::move(done))
{
    VLOG(2) << "Created iteration " << m_impl.graph_id_;
}

IterTask::~IterTask() = default;

uint64_t IterTask::graphId() const
{
    return m_impl.graph_id_;
}

bool IterTask::isCanceled() const
{
    return m_cm.IsCancelled();
}

ResStats IterTask::estimatedPeakAllocation(const DeviceSpec &dev) const
{
    UNUSED(dev);

    return m_impl.cost_mgr_.getForIteration();
}

bool IterTask::prepare()
{
    auto rm = estimatedPeakAllocation(devices::GPU0);
    auto ectx = m_impl.params_.ins;
    return ectx->m_item->beginIteration(ectx->m_ticket, rm, graphId());
}

void IterTask::runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept
{
    std::shared_ptr<IterationContext> sictx(std::move(ictx));
    sictx->setGraphId(graphId());

    (new ExecutorState(m_args, &m_impl))->RunAsync(m_done, sictx);
}

void IterTask::cancel()
{
    m_cm.StartCancel();
}

} // namespace salus::oplib::tensorflow
