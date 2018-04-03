/*
 * <one line to give the scheduler's name and an idea of what it does.>
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

#include "basescheduler.h"

#include "execution/operationtask.h"
#include "execution/scheduler/operationitem.h"
#include "utils/threadutils.h"
#include "utils/macros.h"
#include "utils/envutils.h"
#include "platform/logging.h"

namespace {
bool useGPU()
{
    auto use = sstl::fromEnvVar("EXEC_SCHED_USE_GPU", true);
    VLOG(2) << "Scheduling using: " << (use ? "GPU,CPU" : "CPU");
    return use;
}
} // namespace

using namespace salus;

SchedulerRegistary &SchedulerRegistary::instance()
{
    static SchedulerRegistary registary;
    return registary;
}

SchedulerRegistary::SchedulerRegistary() = default;

SchedulerRegistary::~SchedulerRegistary() = default;

SchedulerRegistary::Register::Register(std::string_view name, SchedulerFactory factory)
{
    auto &registary = SchedulerRegistary::instance();
    sstl::Guard guard(registary.m_mu);
    auto [iter, inserted] = registary.m_schedulers.try_emplace(std::string(name), std::move(factory));
    UNUSED(iter);
    if (!inserted) {
        LOG(FATAL) << "Duplicate registration of execution scheduler under name " << name;
    }
}

std::unique_ptr<BaseScheduler> SchedulerRegistary::create(std::string_view name, ExecutionEngine &engine) const
{
    sstl::Guard guard(m_mu);
    auto iter = m_schedulers.find(name);
    if (iter == m_schedulers.end()) {
        LOG(ERROR) << "No scheduler registered under name: " << name;
        return nullptr;
    }
    return iter->second.factory(engine);
}

BaseScheduler::BaseScheduler(ExecutionEngine &engine) : m_engine(engine) {}

BaseScheduler::~BaseScheduler() = default;

void BaseScheduler::notifyPreSchedulingIteration(const SessionList &sessions, const SessionChangeSet &changeset,
                                                 sstl::not_null<CandidateList *> candidates)
{
    UNUSED(sessions);
    UNUSED(changeset);
    UNUSED(candidates);

    sstl::Guard g(m_muRes);
    m_missingRes.clear();
}

bool BaseScheduler::maybePreAllocateFor(OperationItem &opItem, const DeviceSpec &spec)
{
    auto item = opItem.sess.lock();
    if (!item) {
        return false;
    }

    auto usage = opItem.op->estimatedUsage(spec);

    // TODO: use an algorithm to decide streams
    if (spec.type == DeviceType::GPU) {
        usage[{ResourceType::GPU_STREAM, spec}] = 1;
    }

    Resources missing;
    auto rctx = m_engine.makeResourceContext(item, spec, usage, &missing);
    if (!rctx->isGood()) {
        // Failed to pre allocate resources
        sstl::Guard g(m_muRes);
        m_missingRes.emplace(&opItem, std::move(missing));
        return false;
    }

    auto ticket = rctx->ticket();
    if (!opItem.op->prepare(std::move(rctx))) {
        return false;
    }

    sstl::Guard g(item->tickets_mu);
    item->tickets.insert(ticket);
    return true;
}

bool BaseScheduler::insufficientMemory(const DeviceSpec &spec)
{
    sstl::Guard g(m_muRes);

    if (m_missingRes.empty()) {
        return false;
    }

    // we need paging if all not scheduled opItems in this iteration
    // are missing memory resource on the device
    for (const auto &[pOpItem, missing] : m_missingRes) {
        UNUSED(pOpItem);
        for (const auto &[tag, amount] : missing) {
            UNUSED(amount);
            auto insufficientMemory = tag.type == ResourceType::MEMORY && tag.device == spec;
            if (!insufficientMemory) {
                return false;
            }
        }
    }
    return true;
}

std::string BaseScheduler::debugString(const PSessionItem &item) const
{
    UNUSED(item);
    return {};
}

std::string BaseScheduler::debugString() const
{
    return name();
}

POpItem BaseScheduler::submitTask(POpItem &&opItem)
{
    auto item = opItem->sess.lock();
    if (!item) {
        // session already deleted, discard this task sliently
        return nullptr;
    }

    VLOG(3) << "Scheduling opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
    TIMED_SCOPE_IF(timerInnerObj, "BaseScheduler::submitTask", VLOG_IS_ON(1));

    CVLOG(1, logging::kOpTracing) << "OpItem Event " << opItem->op->DebugString()
                                  << " event: inspected";
    bool scheduled = false;
    DeviceSpec spec{};
    for (auto dt : opItem->op->supportedDeviceTypes()) {
        if (dt == DeviceType::GPU && !useGPU()) {
            continue;
        }
        spec.type = dt;
        spec.id = 0;
        if (maybePreAllocateFor(*opItem, spec)) {
            VLOG(3) << "Task scheduled on " << spec.DebugString();
            scheduled = true;
            break;
        }
    }

    CVLOG(1, logging::kOpTracing) << "OpItem Event " << opItem->op->DebugString()
                                  << " event: prealloced";

    // Send to thread pool
    if (scheduled) {
        opItem = m_engine.submitTask(std::move(opItem));
    } else {
        VLOG(2) << "Failed to schedule opItem in session " << item->sessHandle << ": "
                << opItem->op->DebugString();
    }
    return opItem;
}

size_t BaseScheduler::submitAllTaskFromQueue(const PSessionItem &item)
{
    auto &queue = item->bgQueue;
    size_t scheduled = 0;

    if (queue.empty()) {
        return scheduled;
    }

    // Exam if queue front has been waiting for a long time
    if (item->holWaiting > m_engine.schedulingParam().maxHolWaiting) {
        VLOG(2) << "In session " << item->sessHandle << ": HOL waiting exceeds maximum: " << item->holWaiting
                << " (max=" << m_engine.schedulingParam().maxHolWaiting << ")";
        // Only try to schedule head in this case
        auto &head = queue.front();
        head = submitTask(std::move(head));
        if (!head) {
            queue.pop_front();
            scheduled += 1;
        }
    } else {
        auto size = queue.size();
        SessionItem::UnsafeQueue stage;
        stage.swap(queue);

#if defined(SALUS_ENABLE_PARALLEL_SCHED)
        // Do all schedule in queue in parallel
        std::vector<std::future<std::shared_ptr<OperationItem>>> futures;
        futures.reserve(stage.size());
        for (auto &opItem : stage) {
            auto fu = m_engine.pool().post([opItem = std::move(opItem), this]() mutable {
                DCHECK(opItem);
                return submitTask(std::move(opItem));
            });
            futures.emplace_back(std::move(fu));
        }

        for (auto &fu : futures) {
            auto poi = fu.get();
            if (poi) {
                queue.emplace_back(std::move(poi));
            }
        }
#else
        for (auto &opItem : stage) {
            auto poi = submitTask(std::move(opItem));
            if (poi) {
                queue.emplace_back(std::move(poi));
            }
        }
#endif
        VLOG(2) << "All opItem in session " << item->sessHandle << " examined";

        scheduled = size - queue.size();
    }

    // update queue head waiting
    if (queue.empty()) {
        item->queueHeadHash = 0;
        item->holWaiting = 0;
    } else if (queue.front()->hash() == item->queueHeadHash) {
        item->holWaiting += scheduled;
    } else {
        item->queueHeadHash = queue.front()->hash();
        item->holWaiting = 0;
    }

    return scheduled;
}
