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

#include "executor.pb.h"

#include <memory>
#include <unordered_map>

typedef std::unique_ptr<google::protobuf::Message> ProtoPtr;

class ITask
{
public:
    virtual ~ITask();

    virtual bool prepare(DeviceType dev);

    virtual ProtoPtr run() = 0;

    template<typename RESPONSE>
    std::unique_ptr<RESPONSE> run()
    {
        return utils::static_unique_ptr_cast<RESPONSE, google::protobuf::Message>(run());
    }
};

/**
 * @todo write docs
 */
class IOpLibrary
{
public:
    virtual ~IOpLibrary();

    virtual bool accepts(const executor::OpKernelDef &operation) = 0;

    virtual std::unique_ptr<ITask> createRunTask(const executor::OpKernelDef &opeartion,
                                                 const executor::OpContextDef &context) = 0;

    virtual std::unique_ptr<ITask> createFetchTask(const executor::FetchRequest &fetch) = 0;
    virtual std::unique_ptr<ITask> createPushTask(const executor::PushRequest &push) = 0;
};

class OpLibraryRegistary final
{
public:
    OpLibraryRegistary();

    ~OpLibraryRegistary() = default;

    void registerOpLibrary(executor::OpLibraryType libraryType, std::unique_ptr<IOpLibrary> &&library);

    IOpLibrary *findOpLibrary(const executor::OpLibraryType libraryType) const;
    IOpLibrary *findSuitableOpLibrary(const executor::OpKernelDef &opdef) const;

    static OpLibraryRegistary &instance();

private:
    std::unordered_map<executor::OpLibraryType, std::unique_ptr<IOpLibrary>> m_opLibraries;
};

#endif // IOPLIBRARY_H
