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

#include "pack.h"

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
SchedulerRegistary::Register reg("pack", [](auto &engine) {
    return std::make_unique<PackScheduler>(engine);
});
} // namespace

PackScheduler::PackScheduler(ExecutionEngine &engine) : IScheduler(engine) {}

PackScheduler::~PackScheduler() = default;

std::string PackScheduler::name() const
{
    return "pack";
}

void PackScheduler::selectCandidateSessions(const SessionList &sessions,
                                            const SessionChangeSet &changeset,
                                            utils::not_null<CandidateList*> candidates)
{
    UNUSED(changeset);

    candidates->clear();
    for (auto &sess : sessions) {
        candidates->emplace_back(sess);
    }
}

std::pair<size_t, bool> PackScheduler::maybeScheduleFrom(PSessionItem item)
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
