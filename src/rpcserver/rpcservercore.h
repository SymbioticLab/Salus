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

#include "utils/protoutils.h"

#include <memory>

namespace executor {
class PushRequest;
class PushResponse;
class FetchRequest;
class FetchResponse;
class RunRequest;
class RunResponse;
class AllocRequest;
class AllocResponse;
class DeallocRequest;
class DeallocResponse;
}

#define CALL_ALL_SERVICE_NAME(m) \
    m(Run) \
    m(Fetch) \
    m(Push) \
    m(Alloc) \
    m(Dealloc)

/**
 * @todo write docs
 */
class RpcServerCore
{
public:
    /**
     * Dispatch the call.
     *
     * @return the response protobuf message, gauranteed to be not null.
     */
    ProtoPtr dispatch(const std::string &type, const ::google::protobuf::Message *request);

private:

#define DECL_METHOD(name) \
    std::unique_ptr<executor:: name##Response> name (const executor:: name##Request *request);

    CALL_ALL_SERVICE_NAME(DECL_METHOD)

#undef DECL_METHOD
};

#endif // RPCSERVER_H
