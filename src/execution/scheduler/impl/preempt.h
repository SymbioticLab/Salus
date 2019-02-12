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

#ifndef SALUS_EXEC_SCHED_PREEMPT_H
#define SALUS_EXEC_SCHED_PREEMPT_H

#include "execution/scheduler/basescheduler.h"

#include <chrono>

/**
 * @brief Always gives new session higher priority
 */
class PreemptScheduler : public BaseScheduler
{
public:
    explicit PreemptScheduler(salus::TaskExecutor &engine);
    ~PreemptScheduler() override;

    std::string name() const override;

    void notifyPreSchedulingIteration(const SessionList &sessions,
                                      const SessionChangeSet &changeset,
                                      sstl::not_null<CandidateList *> candidates) override;
    std::pair<size_t, bool> maybeScheduleFrom(PSessionItem item) override;

private:
    std::pair<size_t, bool> reportScheduleResult(size_t scheduled) const;

    std::unordered_map<std::string, int> priorities;
};

#endif // SALUS_EXEC_SCHED_PREEMPT_H
