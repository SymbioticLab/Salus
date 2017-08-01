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
 * 
 */

#include "executionengine.h"

#include "utils/macros.h"
#include "utils/envutils.h"

ExecutionEngine &ExecutionEngine::instance()
{
    static ExecutionEngine eng;
    return eng;
}

ExecutionEngine::ExecutionEngine()
    : m_qscope(q::scoped_initialize())
    , m_qec(q::make_execution_context<q::threadpool,
                                      q::direct_scheduler>("executionengine",
                                                           // The queue passed in here is only used for threadpool
                                                           // termination. We don't care about it. Thus this queue
                                                           // is not connected to any event dispatcher
                                                           q::make_shared<q::queue>(0)))
{

}

ExecutionEngine::~ExecutionEngine() = default;

namespace {
bool useGPU()
{
    auto use = utils::fromEnvVar("EXEC_SCHED_USE_GPU", true);
    INFO("Scheduling using: {}", use ? "GPU,CPU" : "CPU");
    return use;
}

} // namespace

bool ExecutionEngine::schedule(ITask *t)
{
    // TODO: implement device selection

    if (useGPU()) {
        if (trySchedule(t, DeviceType::GPU)) {
            INFO("Task scheduled on GPU");
            return true;
        }
    }

    if (trySchedule(t, DeviceType::CPU)) {
        INFO("Task scheduled on CPU");
        return true;
    }
    return false;
}

bool ExecutionEngine::trySchedule(ITask *t, const DeviceSpec &dev)
{
    auto expectedDev = dev;
    if (t->prepare(expectedDev)) {
        return true;
    }

    if (expectedDev != dev) {
        // the task wants to run on a different device
        return t->prepare(expectedDev);
    }
    return false;
}
