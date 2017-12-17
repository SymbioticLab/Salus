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

#include "fairscheduler.h"

#include "execution/scheduler/operationitem.h"
#include "execution/operationtask.h"
#include "utils/macros.h"
#include "utils/date.h"
#include "platform/logging.h"
#include "utils/envutils.h"

#include <chrono>

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
using FpSeconds = std::chrono::duration<double, seconds::period>;
using namespace std::chrono_literals;
using namespace date;

namespace {
SchedulerRegistary::Register reg("fair", [](auto &engine) {
    return std::make_unique<FairScheduler>(engine);
});

bool useGPU()
{
    auto use = utils::fromEnvVar("EXEC_SCHED_USE_GPU", true);
    VLOG(2) << "Scheduling using: " << (use ? "GPU,CPU" : "CPU");
    return use;
}

} // namespace

FairScheduler::FairScheduler(ExecutionEngine &engine) : IScheduler(engine) {}

FairScheduler::~FairScheduler() = default;

void FairScheduler::selectCandidateSessions(const SessionList &sessions,
                                            const SessionChangeSet &changeset,
                                            boost::container::small_vector_base<PSessionItem> *candidates)
{
    static auto lastSnapshotTime = system_clock::now();

    DCHECK_NOTNULL(candidates);

    candidates->clear();

    // Snapshot resource usage counter first, or reset them
    auto now = system_clock::now();
    auto sSinceLastSnapshot = FpSeconds(now - lastSnapshotTime).count();
    lastSnapshotTime = now;
    for (auto &sess : sessions) {
        if (changeset.numSessionAdded == 0) {
            // calculate progress counter increase since last snapshot
            size_t mem = sess->resourceUsage(ResourceTag::GPU0Memory());
            sess->unifiedResSnapshot += mem * sSinceLastSnapshot;
        } else {
            sess->unifiedResSnapshot = 0;
        }

        candidates->emplace_back(sess);
    }

    // Sort sessions if needed. We assume m_sessions.size() is always no more than a few,
    // therefore sorting in every iteration is acceptable.
    if (changeset.numSessionAdded == 0 && m_engine.schedulingParam().useFairnessCounter) {
        using std::sort;
        sort(candidates->begin(), candidates->end(), [](const auto &lhs, const auto &rhs) {
            return lhs->unifiedResSnapshot < rhs->unifiedResSnapshot;
        });
    }
}

std::pair<size_t, bool> FairScheduler::maybeScheduleFrom(PSessionItem item)
{
    auto &queue = item->bgQueue;
    size_t scheduled = 0;

    auto size = queue.size();
    VLOG(3) << "Scheduling all opItem in session " << item->sessHandle << ": queue size " << size;
    if (size == 0) {
        return reportScheduleResult(scheduled);
    }

    if (item->forceEvicted) {
        // cancel all pending tasks
        scheduled = size;
        for (auto &opItem : queue) {
            opItem->op->cancel();
        }
        return reportScheduleResult(scheduled);
    }

    // Exam if queue front has been waiting for a long time
    if (item->holWaiting > m_engine.schedulingParam().maxHolWaiting) {
        VLOG(2) << "In session " << item->sessHandle << ": HOL waiting exceeds maximum: " << item->holWaiting
                << " (max=" << m_engine.schedulingParam().maxHolWaiting << ")";
        // Only try to schedule head in this case
        auto &head = queue.front();
        head = scheduleTask(std::move(head));
        if (!head) {
            queue.pop_front();
            scheduled += 1;
        }
    } else {
        // Do all schedule in queue in parallel
        SessionItem::UnsafeQueue stage;
        stage.swap(queue);

        std::vector<std::future<std::shared_ptr<OperationItem>>> futures;
        futures.reserve(stage.size());
        for (auto &opItem : stage) {
            auto fu = m_engine.pool().post([opItem = std::move(opItem), this]() mutable {
                DCHECK(opItem);
                return scheduleTask(std::move(opItem));
            });
            futures.emplace_back(std::move(fu));
        }

        VLOG(2) << "All opItem in session " << item->sessHandle << " examined";

        for (auto &fu : futures) {
            auto poi = fu.get();
            if (poi) {
                queue.emplace_back(std::move(poi));
            }
        }

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

    return reportScheduleResult(scheduled);
}

POpItem FairScheduler::scheduleTask(POpItem &&opItem)
{
    auto item = opItem->sess.lock();
    if (!item) {
        // session already deleted, discard this task sliently
        return nullptr;
    }

    VLOG(3) << "Scheduling opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
    TIMED_SCOPE_IF(timerInnerObj, "FairScheduler::scheduleTask", VLOG_IS_ON(1));

    opItem->tInspected = system_clock::now();
    bool scheduled = false;
    DeviceSpec spec;
    for (auto dt : opItem->op->supportedDeviceTypes()) {
        if (dt == DeviceType::GPU && !useGPU()) {
            continue;
        }
        spec = {dt, 0};
        if (maybePreAllocateFor(*opItem, spec)) {
            VLOG(3) << "Task scheduled on " << spec.DebugString();
            scheduled = true;
            break;
        }
    }

    // Send to thread pool
    if (scheduled) {
        opItem = submitTask(std::move(opItem));
    } else {
        VLOG(2) << "Failed to schedule opItem in session " << item->sessHandle << ": "
                << opItem->op->DebugString();
    }
    return opItem;
}

std::pair<size_t, bool> FairScheduler::reportScheduleResult(size_t scheduled) const
{
    static auto workConservative = m_engine.schedulingParam().workConservative;
    // make sure the first session (with least progress) is
    // get scheduled solely, thus can keep up, without other
    // sessions interfere
    return {scheduled, workConservative && scheduled == 0};
}
