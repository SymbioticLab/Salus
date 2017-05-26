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

#include <memory>

namespace google {
namespace protobuf {
class Message;
}
}

namespace executor {
class RunRequest;
class RunResponse;
class AllocRequest;
class AllocResponse;
class DeallocRequest;
class DeallocResponse;
}

typedef std::unique_ptr<::google::protobuf::Message> ProtoPtr;
/**
 * @todo write docs
 */
class RpcServerCore
{
public:
    /**
     * Create the protobuf message of specific type name `type` from a byte buffer `data` of length `len`.
     *
     * @return created Message, or nullptr if specified type not found.
     */
    ProtoPtr createMessage(const std::string type, const void *data, size_t len);

    /**
     * Dispatch the call.
     *
     * @return the response protobuf message. Gauranteed to be not null.
     */
    ProtoPtr dispatch(const std::string &type, const ::google::protobuf::Message *request);

    void Run(const executor::RunRequest *request, executor::RunResponse *response);
    void Alloc(const executor::AllocRequest *request, executor::AllocResponse *response);
    void Dealloc(const executor::DeallocRequest *request, executor::DeallocResponse *response);

private:
};

#endif // RPCSERVER_H
