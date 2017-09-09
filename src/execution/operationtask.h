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

#ifndef OPEARTIONTASK_H
#define OPEARTIONTASK_H

#include "execution/devices.h"
#include "execution/resources.h"

class OperationTask
{
public:
    virtual ~OperationTask();

    virtual bool prepare(DeviceSpec &dev) = 0;
    virtual void run() = 0;

    // estimate usage and cache the result
    virtual ResourceMap estimatedUsage(const DeviceSpec &dev) = 0;

    // Returns cached usage
    virtual bool lastUsage(const DeviceSpec &dev, ResourceMap &res) = 0;
};

#endif // OPEARTIONTASK_H
