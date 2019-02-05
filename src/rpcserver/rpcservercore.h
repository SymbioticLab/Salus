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

#ifndef RPCSERVER_H
#define RPCSERVER_H

#include "zmqserver.h"

#include "utils/protoutils.h"

#include <memory>

namespace executor {
class RunRequest;
class RunResponse;
class RunGraphRequest;
class RunGraphResponse;
class AllocRequest;
class AllocResponse;
class DeallocRequest;
class DeallocResponse;
class CustomRequest;
class CustomResponse;
class EvenlopDef;
} // namespace executor

#define CALL_ALL_SERVICE_NAME(m) \
    m(Run) \
    m(RunGraph) \
    m(Alloc) \
    m(Dealloc) \
    m(Custom)

class IOpLibrary;
/**
 * @todo write docs
 */
class RpcServerCore
{
public:
    RpcServerCore();

    ~RpcServerCore();
    /**
     * Dispatch the call.
     */
    void dispatch(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                  const ::google::protobuf::Message &request);

private:
#define DECL_METHOD(name)                                                                                    \
    void name(ZmqServer::Sender &&sender, IOpLibrary *oplib, const executor::EvenlopDef &evenlop,            \
              const executor::name##Request &request);

    CALL_ALL_SERVICE_NAME(DECL_METHOD)

#undef DECL_METHOD
};

#endif // RPCSERVER_H
