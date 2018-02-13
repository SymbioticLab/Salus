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

#include "preempt.h"

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
SchedulerRegistary::Register reg("preempt", [](auto &engine) {
    return std::make_unique<PreemptScheduler>(engine);
});
} // namespace

PreemptScheduler::PreemptScheduler(ExecutionEngine &engine) : IScheduler(engine) {}

PreemptScheduler::~PreemptScheduler() = default;

std::string PreemptScheduler::name() const
{
    return "preempt";
}

void PreemptScheduler::selectCandidateSessions(const SessionList &sessions,
                                               const SessionChangeSet &changeset,
                                               sstl::not_null<CandidateList*> candidates)
{
    static int priorityCounter = 0;

    candidates->clear();

    // newly added session has higher priority and preempts other sessions.
    if (changeset.numAddedSessions != 0) {
        for (auto it = changeset.addedSessionBegin; it != changeset.addedSessionEnd; ++it) {
            priorities[(*it)->sessHandle] = priorityCounter;
        }
        ++priorityCounter;
    }

    for (auto &sess : sessions) {
        candidates->emplace_back(sess);
    }

    // Sort sessions by priority. We assume m_sessions.size() is always no more than a few,
    // therefore sorting in every iteration is acceptable.
    using std::sort; // sort is asc order is using operator<.
    sort(candidates->begin(), candidates->end(), [this](const auto &lhs, const auto &rhs) {
        return priorities[lhs->sessHandle] > priorities[rhs->sessHandle];
    });
}

std::pair<size_t, bool> PreemptScheduler::maybeScheduleFrom(PSessionItem item)
{
    auto scheduled = submitAllTaskFromQueue(item);

    return reportScheduleResult(scheduled);
}

std::pair<size_t, bool> PreemptScheduler::reportScheduleResult(size_t scheduled) const
{
    static auto workConservative = m_engine.schedulingParam().workConservative;
    // make sure the first session (with least progress) is
    // get scheduled solely, thus can keep up, without other
    // sessions interfere
    return {scheduled, workConservative && scheduled == 0};
}
