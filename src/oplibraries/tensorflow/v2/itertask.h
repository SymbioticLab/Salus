//
// Created by peifeng on 4/16/18.
//

#ifndef SALUS_OPLIB_TENSORFLOW_ITERTASK_H
#define SALUS_OPLIB_TENSORFLOW_ITERTASK_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "execution/iterationtask.h"
#include "oplibraries/tensorflow/v2/md_executor_impl.h"

#include <atomic>

namespace salus::oplib::tensorflow {
class IterTask : public IterationTask
{
    static std::atomic_int_fast64_t NextSeq;

    ExecutorImpl &m_impl;

    std::string m_name;

    tf::CancellationManager &m_cm;

    tf::Executor::Args m_args;
    /**
     * @brief Callback for ExecutorImpl
     */
    tf::Executor::DoneCallback m_done;

public:
    IterTask(ExecutorImpl &impl, const tf::Executor::Args &args, tf::Executor::DoneCallback done);

    ~IterTask() override;

    const std::string &name() const override;

    bool prepare() override;

    ResStats estimatedPeakAllocation(const DeviceSpec &dev) const override;

    void runAsync(std::shared_ptr<IterationContext> &&ictx) noexcept override;

    void cancel() override;

    bool isCanceled() const override;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_ITERTASK_H
