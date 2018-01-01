/*
 * <one line to give the scheduler's name and an idea of what it does.>
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

#include "ischeduler.h"

#include "execution/operationtask.h"
#include "execution/scheduler/operationitem.h"
#include "utils/threadutils.h"
#include "utils/macros.h"
#include "utils/envutils.h"
#include "platform/logging.h"

#include <chrono>

using std::chrono::system_clock;

namespace {
bool useGPU()
{
    auto use = utils::fromEnvVar("EXEC_SCHED_USE_GPU", true);
    VLOG(2) << "Scheduling using: " << (use ? "GPU,CPU" : "CPU");
    return use;
}
} // namespace

SchedulerRegistary &SchedulerRegistary::instance()
{
    static SchedulerRegistary registary;
    return registary;
}

SchedulerRegistary::SchedulerRegistary() = default;

SchedulerRegistary::~SchedulerRegistary() = default;

SchedulerRegistary::Register::Register(std::string_view name, SchedulerFactory factory)
{
    auto &registary = SchedulerRegistary::instance();
    utils::Guard guard(registary.m_mu);
    auto [iter, inserted] = registary.m_schedulers.try_emplace(std::string(name), std::move(factory));
    UNUSED(iter);
    if (!inserted) {
        LOG(FATAL) << "Duplicate registration of execution scheduler under name " << name;
    }
}

std::unique_ptr<IScheduler> SchedulerRegistary::create(std::string_view name, ExecutionEngine &engine) const
{
    utils::Guard guard(m_mu);
    auto iter = m_schedulers.find(name);
    if (iter == m_schedulers.end()) {
        LOG(ERROR) << "No scheduler registered under name: " << name;
        return nullptr;
    }
    return iter->second.factory(engine);
}

IScheduler::IScheduler(ExecutionEngine &engine) : m_engine(engine) {}

IScheduler::~IScheduler() = default;

bool IScheduler::maybePreAllocateFor(OperationItem &opItem, const DeviceSpec &spec)
{
    return m_engine.maybePreAllocateFor(opItem, spec);
}

POpItem IScheduler::submitTask(POpItem &&opItem)
{
    auto item = opItem->sess.lock();
    if (!item) {
        // session already deleted, discard this task sliently
        return nullptr;
    }

    VLOG(3) << "Scheduling opItem in session " << item->sessHandle << ": " << opItem->op->DebugString();
    TIMED_SCOPE_IF(timerInnerObj, "IScheduler::submitTask", VLOG_IS_ON(1));

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
        opItem = m_engine.submitTask(std::move(opItem));
    } else {
        VLOG(2) << "Failed to schedule opItem in session " << item->sessHandle << ": "
                << opItem->op->DebugString();
    }
    return opItem;
}
