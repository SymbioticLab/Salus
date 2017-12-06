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
