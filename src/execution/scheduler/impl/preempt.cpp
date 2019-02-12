/*
 * Copyright 2019 Peifeng Yu <peifeng@umich.edu>
 * 
 * This file is part of Salus
 * (see https://github.com/SymbioticLab/Salus).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
using namespace salus;

namespace {
SchedulerRegistary::Register reg("preempt", [](auto &engine) {
    return std::make_unique<PreemptScheduler>(engine);
});
// HACK
SchedulerRegistary::Register reg2("rr", [](auto &engine) {
    return std::make_unique<PreemptScheduler>(engine);
});
SchedulerRegistary::Register reg3("fifo", [](auto &engine) {
    return std::make_unique<PreemptScheduler>(engine);
});
} // namespace

PreemptScheduler::PreemptScheduler(TaskExecutor &engine) : BaseScheduler(engine) {}

PreemptScheduler::~PreemptScheduler() = default;

std::string PreemptScheduler::name() const
{
    return "preempt";
}

void PreemptScheduler::notifyPreSchedulingIteration(const SessionList &sessions,
                                                    const SessionChangeSet &changeset,
                                                    sstl::not_null<CandidateList *> candidates)
{
    static int priorityCounter = 0;

    BaseScheduler::notifyPreSchedulingIteration(sessions, changeset, candidates);

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
    static auto workConservative = m_taskExec.schedulingParam().workConservative;
    // make sure the first session (with least progress) is
    // get scheduled solely, thus can keep up, without other
    // sessions interfere
    return {scheduled, workConservative && scheduled == 0};
}
