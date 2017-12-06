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

#include "rpcserver/zmqserver.h"
#include "utils/macros.h"
#include "utils/pointerutils.h"

#include "protos.h"

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

/**
 * @todo write docs
 */
class IOpLibrary
{
public:
    virtual ~IOpLibrary();

    /**
     * Any non trival initialization and cleanup should be done
     * in the following methods. Constructor and destructor are
     * called before/after main(), thus certain system is not available
     */
    virtual bool initialize() = 0;
    virtual void uninitialize() = 0;

    virtual bool accepts(const executor::OpKernelDef &operation) = 0;

    using DoneCallback = std::function<void(ProtoPtr &&)>;

    virtual void onRun(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                       const executor::RunRequest &request, DoneCallback cb) = 0;

    virtual void onRunGraph(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                            const executor::RunGraphRequest &request, DoneCallback cb) = 0;

    virtual void onCustom(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                          const executor::CustomRequest &msg, DoneCallback cb) = 0;
};

class OpLibraryRegistary final
{
public:
    OpLibraryRegistary();

    ~OpLibraryRegistary() = default;

    struct Register
    {
        Register(executor::OpLibraryType libraryType, std::unique_ptr<IOpLibrary> &&library,
                 int priority = 50);
    };
    void registerOpLibrary(executor::OpLibraryType libraryType, std::unique_ptr<IOpLibrary> &&library,
                           int priority);

    void initializeLibraries();
    void uninitializeLibraries();

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
    bool initialized;
};

#endif // IOPLIBRARY_H
