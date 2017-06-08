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

#ifndef EXECUTIONENGINE_H
#define EXECUTIONENGINE_H

#include "devices.h"

#include "oplibraries/ioplibrary.h"

#include <memory>
#include <future>

/**
 * @todo write docs
 */
class ExecutionEngine
{
public:
    static ExecutionEngine &instance();

    ~ExecutionEngine();

    template<typename ResponseType>
    std::future<std::unique_ptr<ResponseType>> enqueue(std::unique_ptr<ITask> &&task)
    {
        typedef std::unique_ptr<ResponseType> PResponse;

        std::promise<PResponse> p;

        if (task->prepare(DeviceType::CPU)) {
            p.set_value(task->run<ResponseType>());
        } else {
            p.set_value({});
        }

        return p.get_future();
    }

private:
    ExecutionEngine();
};

#endif // EXECUTIONENGINE_H
