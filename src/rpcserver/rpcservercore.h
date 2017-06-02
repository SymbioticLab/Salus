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

    void Run(const executor::RunRequest *request, executor::RunResponse *response);
    void Fetch(const executor::FetchRequest *request, executor::FetchResponse *response);

    void Push(const executor::PushRequest *request, executor::PushResponse *response);

    void Alloc(const executor::AllocRequest *request, executor::AllocResponse *response);
    void Dealloc(const executor::DeallocRequest *request, executor::DeallocResponse *response);

private:
};

#endif // RPCSERVER_H
