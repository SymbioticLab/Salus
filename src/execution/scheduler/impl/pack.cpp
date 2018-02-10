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
                                            salus::not_null<CandidateList*> candidates)
{
    UNUSED(changeset);

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
