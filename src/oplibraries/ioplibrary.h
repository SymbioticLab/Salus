/*
 * Copyright 2019 Peifeng Yu <peifeng@umich.edu>
 * 
 * This file is part of Salus
 * (see https://github.com/SymbioticLab/Salus).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

    ~OpLibraryRegistary();

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
    int initialized;
};

#endif // IOPLIBRARY_H
