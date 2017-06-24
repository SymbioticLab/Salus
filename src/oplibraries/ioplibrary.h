/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017 Peifeng Yu <peifeng@umich.edu>
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

#ifndef IOPLIBRARY_H
#define IOPLIBRARY_H

#include "execution/devices.h"
#include "utils/pointerutils.h"
#include "rpcserver/zmqserver.h"

#include "executor.pb.h"

#include <memory>
#include <unordered_map>
#include <functional>

typedef std::unique_ptr<google::protobuf::Message> ProtoPtr;

class ITask
{
public:
    virtual ~ITask();

    virtual bool prepare(DeviceType dev);

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

template<typename Fn>
class LambdaTask : public ITask
{
public:
    explicit LambdaTask(Fn &&fn) : m_fn(std::move(fn)) {}

    ProtoPtr run() override
    {
        return m_fn();
    }

private:
    Fn m_fn;
};

template<typename Fn>
PTask make_lambda_task(Fn fn)
{
    return std::make_unique<LambdaTask<Fn>>(std::move(fn));
}

/**
 * @todo write docs
 */
class IOpLibrary
{
public:
    virtual ~IOpLibrary();

    virtual bool accepts(const executor::OpKernelDef &operation) = 0;

    virtual PTask createRunTask(ZmqServer::Sender sender,
                                const executor::EvenlopDef &evenlop,
                                const executor::RunRequest &request) = 0;

    virtual PTask createRunGraphTask(ZmqServer::Sender sender,
                                     const executor::EvenlopDef &evenlop,
                                     const executor::RunGraphRequest &request) = 0;

    virtual PTask createCustomTask(ZmqServer::Sender sender,
                                   const executor::EvenlopDef &evenlop,
                                   const executor::CustomRequest &msg) = 0;
};

class OpLibraryRegistary final
{
public:
    OpLibraryRegistary();

    ~OpLibraryRegistary() = default;

    struct Register
    {
        Register(executor::OpLibraryType libraryType, std::unique_ptr<IOpLibrary> &&library);
    };
    void registerOpLibrary(executor::OpLibraryType libraryType, std::unique_ptr<IOpLibrary> &&library);

    IOpLibrary *findOpLibrary(const executor::OpLibraryType libraryType) const;
    IOpLibrary *findSuitableOpLibrary(const executor::OpKernelDef &opdef) const;

    static OpLibraryRegistary &instance();

private:
    std::unordered_map<executor::OpLibraryType, std::unique_ptr<IOpLibrary>> m_opLibraries;
};

#define REGISTER_OPLIBRARY(type, name) \
    OpLibraryRegistary::Register name ## register ((type), std::make_unique<name>())

#endif // IOPLIBRARY_H
