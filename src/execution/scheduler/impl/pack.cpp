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
using namespace salus;

namespace {
SchedulerRegistary::Register reg("pack", [](auto &engine) {
    return std::make_unique<PackScheduler>(engine);
});

SchedulerRegistary::Register reg2("mix", [](auto &engine) {
    return std::make_unique<PackScheduler>(engine);
});

} // namespace

PackScheduler::PackScheduler(TaskExecutor &engine) : BaseScheduler(engine) {}

PackScheduler::~PackScheduler() = default;

std::string PackScheduler::name() const
{
    return "pack";
}

void PackScheduler::notifyPreSchedulingIteration(const SessionList &sessions,
                                                 const SessionChangeSet &changeset,
                                                 sstl::not_null<CandidateList *> candidates)
{
    BaseScheduler::notifyPreSchedulingIteration(sessions, changeset, candidates);

    candidates->clear();
    for (auto &sess : sessions) {
        candidates->emplace_back(sess);
    }
}

std::pair<size_t, bool> PackScheduler::maybeScheduleFrom(PSessionItem item)
{
    auto scheduled = submitAllTaskFromQueue(item);

    return reportScheduleResult(scheduled);
}
