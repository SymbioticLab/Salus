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
#include "platform/logging.h"
#include "memorymgr/memorymgr.h"

#include "executor.pb.h"

#include <unordered_map>
#include <functional>

using namespace executor;
using ::google::protobuf::Message;

ProtoPtr RpcServerCore::dispatch(const std::string &type, const Message *request)
{
#define ITEM(name) \
        {"executor." #name "Request", [&](const Message *request){ \
            auto response = std::make_unique<name ## Response>(); \
            name (static_cast<const name ## Request*>(request), response.get()); \
            return response; \
        }}

    static std::unordered_map<std::string, std::function<ProtoPtr(const Message*)>> funcs {
        ITEM(Push),
        ITEM(Fetch),
        ITEM(Run),
        ITEM(Alloc),
        ITEM(Dealloc),
    };

#undef ITEM

    assert(funcs.count(type) == 1);

    return funcs[type](request);
}

void RpcServerCore::Push(const executor::PushRequest *request, executor::PushResponse *response)
{
    INFO("Serving PushRequest for oplibrary {}", OpLibraryType_Name(request->oplibrary()));

    auto oplib = OpLibraryRegistary::instance().findOpLibrary(request->oplibrary());
    if (!oplib) {
        ERR("Skipping fetch due to failed to find requested OpLibrary.");
        return;
    }

    auto task = oplib->createPushTask(*request);
    if (!task) {
        ERR("Skipping task due to failed creation.");
        return;
    }

    // run the task right away
    auto res = response->mutable_result();
    *res = task->run(response);

    // TODO: Enqueue the task, and run async
}

void RpcServerCore::Fetch(const FetchRequest *request, FetchResponse *response)
{
    INFO("Serving FetchRequest for oplibrary {}", OpLibraryType_Name(request->oplibrary()));

    auto oplib = OpLibraryRegistary::instance().findOpLibrary(request->oplibrary());
    if (!oplib) {
        ERR("Skipping fetch due to failed to find requested OpLibrary.");
        return;
    }

    auto task = oplib->createFetchTask(*request);
    if (!task) {
        ERR("Skipping task due to failed creation.");
        return;
    }

    // run the task right away
    auto res = response->mutable_result();
    *res = task->run(response);

    // TODO: Enqueue the task, and run async
}

void RpcServerCore::Run(const RunRequest *request, RunResponse *response)
{
    const auto &opdef = request->opkernel();
    const auto &ctxdef = request->context();

    INFO("Serving RunRequest with opkernel id {} and oplibrary {}",
         opdef.id(), OpLibraryType_Name(opdef.oplibrary()));

    auto oplib = OpLibraryRegistary::instance().findSuitableOpLibrary(opdef);
    if (!oplib) {
        ERR("Skipping task due to failed to find suitable OpLibrary.");
        return;
    }
    assert(oplib->accepts(opdef));

    auto task = oplib->createRunTask(opdef, ctxdef);
    if (!task) {
        ERR("Skipping task due to failed creation.");
        return;
    }

    // run the task right away
    auto res = response->mutable_result();
    *res = task->run(response->mutable_context());

    // TODO: Enqueue the task, and run async
}

void RpcServerCore::Alloc(const AllocRequest *request, AllocResponse *response)
{
    auto alignment = request->alignment();
    auto num_bytes = request->num_bytes();

    INFO("Serving AllocRequest with alignment {} and num_bytes {}", alignment, num_bytes);

    auto ptr = MemoryMgr::instance().allocate(num_bytes, alignment);
    auto addr_handle = reinterpret_cast<uint64_t>(ptr);
    response->set_addr_handle(addr_handle);

    INFO("Allocated address handel: {:x}", addr_handle);

    response->mutable_result()->set_code(0);
}

void RpcServerCore::Dealloc(const DeallocRequest *request, DeallocResponse *response)
{
    auto addr_handle = request->addr_handle();

    INFO("Serving DeallocRequest with address handel: {:x}", addr_handle);

    MemoryMgr::instance().deallocate(reinterpret_cast<void*>(addr_handle));

    response->mutable_result()->set_code(0);
}
