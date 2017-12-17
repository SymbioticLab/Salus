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

#include "utils/threadutils.h"
#include "utils/macros.h"
#include "platform/logging.h"

#include <chrono>

using std::chrono::system_clock;

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
    return m_engine.submitTask(std::move(opItem));
}

