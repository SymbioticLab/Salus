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

#include "itask.h"

#include "utils/macros.h"
#include "platform/logging.h"
#include "utils/threadutils.h"

ITask::~ITask() = default;

bool ITask::prepare(DeviceSpec &dev)
{
    UNUSED(dev);

    return true;
}

bool ITask::isAsync()
{
    return false;
}

void ITask::runAsync(DoneCallback cb)
{
    UNUSED(cb);
    LOG(FATAL) << "ITask::runAsync Not Implemented";
}

AsyncTask::~AsyncTask() = default;

ProtoPtr AsyncTask::run()
{
    utils::semaphore se;
    ProtoPtr resp;
    runAsync([&se, &resp](auto protoptr){
        resp = std::move(protoptr);
        se.notify();
    });
    se.wait();
    return resp;
}
