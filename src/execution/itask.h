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

#ifndef ITASK_H
#define ITASK_H

#include "execution/devices.h"
#include "utils/pointerutils.h"
#include "utils/protoutils.h"

#include <memory>
#include <functional>

class ITask
{
public:
    virtual ~ITask();

    /**
     * Prepare the task on device dev. The task may propose to run on different device
     * by returning false and modifying dev.
     */
    virtual bool prepare(DeviceSpec &dev);

    virtual bool isAsync();

    virtual ProtoPtr run() = 0;

    template<typename RESPONSE>
    std::unique_ptr<RESPONSE> run()
    {
        return utils::static_unique_ptr_cast<RESPONSE, google::protobuf::Message>(run());
    }

    using DoneCallback = std::function<void(ProtoPtr&&)>;
    virtual void runAsync(DoneCallback cb);

    template<typename RESPONSE>
    void runAsync(std::function<void(std::unique_ptr<RESPONSE>&&)> cb)
    {
        runAsync([cb = std::move(cb)](ProtoPtr &&ptr) mutable {
            cb(utils::static_unique_ptr_cast<RESPONSE, google::protobuf::Message>(std::move(ptr)));
        });
    }
};

using PTask = std::unique_ptr<ITask>;

class AsyncTask : public ITask
{
public:
    ~AsyncTask() override;

    bool isAsync() override { return true; }

    ProtoPtr run() override;
};

template<typename Fn>
class AsyncLambdaTask : public AsyncTask
{
public:
    explicit AsyncLambdaTask(Fn &&fn) : m_fn(std::move(fn)) {}

    void runAsync(DoneCallback cb) override
    {
        m_fn(std::move(cb));
    }

private:
    Fn m_fn;
};

template<typename Fn>
PTask make_async_lambda_task(Fn&& fn)
{
    return std::make_unique<AsyncLambdaTask<Fn>>(std::move(fn));
}


#endif // ITASK_H
