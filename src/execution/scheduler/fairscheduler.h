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

#ifndef FAIRSCHEDULER_H
#define FAIRSCHEDULER_H

#include "ischeduler.h"

#include <chrono>

/**
 * @todo write docs
 */
class FairScheduler : public IScheduler
{
public:
    FairScheduler(ExecutionEngine &engine);
    ~FairScheduler() override;

    void selectCandidateSessions(const SessionList &sessions,
                                 const SessionChangeSet &changeset,
                                 boost::container::small_vector_base<PSessionItem> *candidates) override;
    std::pair<size_t, bool> maybeScheduleFrom(PSessionItem item) override;

private:
    POpItem scheduleTask(POpItem &&opItem);

    std::pair<size_t, bool> reportScheduleResult(size_t scheduled) const;
};

#endif // FAIRSCHEDULER_H
