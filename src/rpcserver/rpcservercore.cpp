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

#include "rpcservercore.h"

#include "oplibraries/ioplibrary.h"
#include "platform/memory.h"

#include "executor.pb.h"

#include <unordered_map>
#include <functional>

using namespace executor;
using ::google::protobuf::Message;

ProtoPtr RpcServerCore::createMessage(const std::string type, const void* data, size_t len)
{
    auto desc = ::google::protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(type);
    if (!desc) {
        return {};
    }

    auto message = ::google::protobuf::MessageFactory::generated_factory()->GetPrototype(desc)->New();
    if (!message) {
        return {};
    }

    auto ok = message->ParseFromArray(data, len);
    if (!ok) {
        return {};
    }

    return ProtoPtr(message);
}

ProtoPtr RpcServerCore::dispatch(const std::string &type, const Message *request)
{
#define ITEM(name) \
        {"executor." #name "Request", [&](const Message *request){ \
            auto response = std::make_unique<name ## Response>(); \
            name (static_cast<const name ## Request*>(request), response.get()); \
            return response; \
        }}

    static std::unordered_map<std::string, std::function<ProtoPtr(const Message*)>> funcs {
        ITEM(Run),
        ITEM(Alloc),
        ITEM(Dealloc),
    };

#undef ITEM

    assert(funcs.count(type) == 1);

    return funcs[type](request);
}

void RpcServerCore::Run(const RunRequest *request, RunResponse *response)
{
    auto opdef = request->opkernel();
    auto ctxdef = request->context();

    auto oplib = OpLibraryRegistary::instance().findSuitableOpLibrary(opdef);
    assert(oplib->accepts(opdef));

    auto task = oplib->createTask(opdef, ctxdef);

    // run the task right away
    auto res = response->mutable_result();
    auto newctxdef = response->mutable_context();

    *res = task->run();
    *newctxdef = task->contextDef();

    // TODO: Enqueue the task, and run async
}

void RpcServerCore::Alloc(const AllocRequest *request, AllocResponse *response)
{
    auto alignment = request->alignment();
    auto num_bytes = request->num_bytes();

    // TODO: use more appropriate memroy allocation
    uint64_t addr_handle = reinterpret_cast<uint64_t>(mem::alignedAlloc(num_bytes, alignment));
    response->set_addr_handle(addr_handle);

    response->mutable_result()->set_code(0);
}

void RpcServerCore::Dealloc(const DeallocRequest *request, DeallocResponse *response)
{
    auto addr_handle = request->addr_handle();

    mem::alignedFree(reinterpret_cast<void*>(addr_handle));

    response->mutable_result()->set_code(0);
}
