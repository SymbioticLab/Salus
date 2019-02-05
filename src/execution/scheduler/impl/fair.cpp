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

#include "fair.h"

#include "execution/scheduler/operationitem.h"
#include "execution/operationtask.h"
#include "utils/macros.h"
#include "utils/date.h"
#include "platform/logging.h"
#include "utils/envutils.h"

#include <chrono>
#include <sstream>

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
SchedulerRegistary::Register reg("fair", [](auto &engine) {
    return std::make_unique<FairScheduler>(engine);
});

} // namespace

FairScheduler::FairScheduler(TaskExecutor &engine) : BaseScheduler(engine) {}

FairScheduler::~FairScheduler() = default;

std::string FairScheduler::name() const
{
    return "fair";
}

void FairScheduler::notifyPreSchedulingIteration(const SessionList &sessions,
                                                 const SessionChangeSet &changeset,
                                                 sstl::not_null<CandidateList *> candidates)
{
    static auto lastSnapshotTime = system_clock::now();

    BaseScheduler::notifyPreSchedulingIteration(sessions, changeset, candidates);

    candidates->clear();

    // Remove old sessions
    for (auto &sess : changeset.deletedSessions) {
        aggResUsages.erase(sess->sessHandle);
    }

    // Sort sessions if needed. When there is addition, counters are reset thus no need to sort.
    // Snapshot resource usage counter first, or reset them
    if (changeset.numAddedSessions == 0) {
        auto now = system_clock::now();
        auto sSinceLastSnapshot = FpSeconds(now - lastSnapshotTime).count();
        lastSnapshotTime = now;
        for (auto &sess : sessions) {
            candidates->emplace_back(sess);
            // calculate progress counter increase since last snapshot
            size_t mem = sess->resourceUsage(resources::GPU0Memory);
            aggResUsages[sess->sessHandle] += mem * sSinceLastSnapshot;
        }

        // We assume m_sessions.size() is always no more than a few,
        // therefore sorting in every iteration is acceptable.
        using std::sort;
        sort(candidates->begin(), candidates->end(), [this](const auto &lhs, const auto &rhs) {
            return aggResUsages.at(lhs->sessHandle) < aggResUsages.at(rhs->sessHandle);
        });
    } else {
        for (auto it = changeset.addedSessionBegin; it != changeset.addedSessionEnd; ++it) {
            LOG(DEBUG) << "Adding session " << (*it)->sessHandle;
        }
        // clear to reset everything to zero.
        aggResUsages.clear();
        aggResUsages.reserve(sessions.size());
        for (auto &sess : sessions) {
            candidates->emplace_back(sess);
            // touch each item once to ensure it's in the map
            aggResUsages[sess->sessHandle] = 0;
        }
    }
}

std::pair<size_t, bool> FairScheduler::maybeScheduleFrom(PSessionItem item)
{
    auto scheduled = submitAllTaskFromQueue(item);

    return reportScheduleResult(scheduled);
}

std::pair<size_t, bool> FairScheduler::reportScheduleResult(size_t scheduled) const
{
    static auto workConservative = m_taskExec.schedulingParam().workConservative;
    // make sure the first session (with least progress) is
    // get scheduled solely, thus can keep up, without other
    // sessions interfere
    return {scheduled, workConservative && scheduled == 0};
}

std::string FairScheduler::debugString(const PSessionItem &item) const
{
    std::ostringstream oss;
    oss << "counter: " << aggResUsages.at(item->sessHandle);
    return oss.str();
}
