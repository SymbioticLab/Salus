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

#ifndef SALUS_OPLIB_TENSORFLOW_EXECTASK_H
#define SALUS_OPLIB_TENSORFLOW_EXECTASK_H

/*
 * Make sure tensorflow_headers is included first before
 * any other headers, so we can correctly override TF logging
 * with ours.
 */
#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "md_executor_impl.h"

#include "execution/operationtask.h"
#include "utils/pointerutils.h"

namespace sstl {
class semaphore;
} // namespace sstl

namespace salus::oplib::tensorflow {

/**
 * @todo write docs
 */
class ExecTask : public OperationTask
{
public:
    ExecTask(ExecutorState *state, sstl::semaphore &num_finished_ops, const ExecutorState::TaggedNode &node,
             const tf::OpKernelContext::Params &initial_params, tf::Rendezvous *rendez, int maxFailures = 2);

    bool prepare(std::unique_ptr<ResourceContext> &&rctx) noexcept override;

    void run(Callbacks cbs) noexcept override;

    void cancel() override;

    int failedTimes() const override
    {
        return failureTimes;
    }

    Resources estimatedUsage(const DeviceSpec &dev) override;

    DeviceTypes supportedDeviceTypes() const override;

    ~ExecTask() override;

    std::string DebugString() override;

    bool isAsync() const override;

    ResourceContext &resourceContext() const override;

private:
    Resources calcUsageFromShape(const DeviceSpec &dev);

    bool maybeMemoryFailure(const tf::Status &s, const MemFailCallback &memFailure);

    void afterCompute(bool is_dead, const Callbacks &cbs);

    void afterRun(const tf::Status &s, const Callbacks &cbs);

    void updateRefEntryTickets(const std::vector<Entry *> &entries);

private:
    // Borrowed from ExecutorState
    ExecutorState *m_state;

    const NodeItem &item;
    tf::Rendezvous *rendez;
    sstl::semaphore &num_finished_ops;

    // Misc
    int failureTimes;
    const int maxFailures;
    Resources failedAlloc;

    // Owned
    ExecutorImpl::DeviceItem ditem;
    tf::Status statusInPrepare;
    ExecutorState::TaggedNode tagged_node;

    POpKernel op_kernel;
    bool kernel_is_async;
    bool has_ref_input;

    std::vector<Entry *> reffedEntries;
    Entry *first_input = nullptr;
    BufferLockVec buflocks;
    uint64_t input_size = 0;

    ExecutorState::TaggedNodeSeq ready;

    // Back storage for params
    TensorValueVec inputs;
    DeviceContextVec input_device_contexts;
    AllocatorAttributeVec input_alloc_attrs;

    // params must out-live pctx
    tf::OpKernelContext::Params params;
    std::unique_ptr<tf::OpKernelContext> pctx;

    // Caches
    std::unordered_map<DeviceSpec, Resources> cachedUsage;
    std::string m_cachedDebugString;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_EXECTASK_H
