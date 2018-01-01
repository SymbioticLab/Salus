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

#include "fair.h"

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

} // namespace

FairScheduler::FairScheduler(ExecutionEngine &engine) : IScheduler(engine) {}

FairScheduler::~FairScheduler() = default;

std::string FairScheduler::name() const
{
    return "fair";
}

void FairScheduler::selectCandidateSessions(const SessionList &sessions,
                                            const SessionChangeSet &changeset,
                                            utils::not_null<CandidateList*> candidates)
{
    static auto lastSnapshotTime = system_clock::now();

    candidates->clear();

    // Snapshot resource usage counter first, or reset them
    auto now = system_clock::now();
    auto sSinceLastSnapshot = FpSeconds(now - lastSnapshotTime).count();
    lastSnapshotTime = now;
    for (auto &sess : sessions) {
        if (changeset.numAddedSessions == 0) {
            // calculate progress counter increase since last snapshot
            size_t mem = sess->resourceUsage(ResourceTag::GPU0Memory());
            sess->unifiedResSnapshot += mem * sSinceLastSnapshot;
        } else {
            sess->unifiedResSnapshot = 0;
        }

        candidates->emplace_back(sess);
    }

    // Sort sessions if needed. When there is addition, counters are reset thus no need to sort.
    if (changeset.numAddedSessions == 0) {
        // We assume m_sessions.size() is always no more than a few,
        // therefore sorting in every iteration is acceptable.
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

    if (queue.empty()) {
        return reportScheduleResult(scheduled);
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
        // Do all schedule in queue in parallel
        auto size = queue.size();
        SessionItem::UnsafeQueue stage;
        stage.swap(queue);

        std::vector<std::future<std::shared_ptr<OperationItem>>> futures;
        futures.reserve(stage.size());
        for (auto &opItem : stage) {
            auto fu = m_engine.pool().post([opItem = std::move(opItem), this]() mutable {
                DCHECK(opItem);
                return submitTask(std::move(opItem));
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

std::pair<size_t, bool> FairScheduler::reportScheduleResult(size_t scheduled) const
{
    static auto workConservative = m_engine.schedulingParam().workConservative;
    // make sure the first session (with least progress) is
    // get scheduled solely, thus can keep up, without other
    // sessions interfere
    return {scheduled, workConservative && scheduled == 0};
}
