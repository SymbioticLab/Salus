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

#include "execution/itask.h"
#include "utils/pointerutils.h"
#include "utils/macros.h"
#include "rpcserver/zmqserver.h"

#include "protos.h"

#include <memory>
#include <unordered_map>
#include <functional>
#include <mutex>

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
        Register(executor::OpLibraryType libraryType, std::unique_ptr<IOpLibrary> &&library, int priority = 50);
    };
    void registerOpLibrary(executor::OpLibraryType libraryType, std::unique_ptr<IOpLibrary> &&library,
                           int priority);

    IOpLibrary *findOpLibrary(const executor::OpLibraryType libraryType) const;
    IOpLibrary *findSuitableOpLibrary(const executor::OpKernelDef &opdef) const;

    static OpLibraryRegistary &instance();

private:
    struct LibraryItem
    {
        std::unique_ptr<IOpLibrary> library;
        int priority;
    };
    mutable std::mutex m_mu;
    std::unordered_map<executor::OpLibraryType, LibraryItem> m_opLibraries;
};

#endif // IOPLIBRARY_H
