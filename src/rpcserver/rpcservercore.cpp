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
#include "execution/executionengine.h"
#include "utils/macros.h"

#include "executor.pb.h"

#include <unordered_map>
#include <functional>

using namespace executor;
using ::google::protobuf::Message;
using std::unique_ptr;

q::promise<ProtoPtr> RpcServerCore::dispatch(ZmqServer::Sender sender, const EvenlopDef &evenlop,
                                             const Message &request)
{
#define ITEM(name) \
        {"executor." #name "Request", [this](auto &&sender, auto *oplib, const auto &request){ \
            return name (std::move(sender), oplib, static_cast<const name ## Request&>(request)) \
                .then([](unique_ptr<name ## Response> &&f){ \
                    return static_cast<ProtoPtr>(std::move(f)); \
            }); \
        }},

    using ServiceMethod = std::function<q::promise<ProtoPtr>(ZmqServer::Sender &&sender, IOpLibrary*, const Message&)>;
    static std::unordered_map<std::string, ServiceMethod> funcs {
        CALL_ALL_SERVICE_NAME(ITEM)
    };

#undef ITEM

    assert(funcs.count(evenlop.type()) == 1);

    INFO("Serving {} for oplibrary {}", evenlop.type(), OpLibraryType_Name(evenlop.oplibrary()));

    auto oplib = OpLibraryRegistary::instance().findOpLibrary(evenlop.oplibrary());
    if (!oplib) {
        ERR("Skipping push due to failed to find requested OpLibrary.");
        return ExecutionEngine::instance().emptyPromise<Message>();
    }

    return funcs[evenlop.type()](std::move(sender), oplib, request);
}

q::promise<unique_ptr<RunResponse>> RpcServerCore::Run(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                                                       const RunRequest &request)
{
    const auto &opdef = request.opkernel();
    const auto &ctxdef = request.context();

    INFO("Serving RunRequest with opkernel id {}", opdef.id());
    assert(oplib->accepts(opdef));

    auto task = oplib->createRunTask(std::move(sender), opdef, ctxdef);
    if (!task) {
        ERR("Skipping task due to failed creation.");
        return ExecutionEngine::instance().emptyPromise<RunResponse>();
    }

    return ExecutionEngine::instance().enqueue<RunResponse>(std::move(task));
}

q::promise<unique_ptr<AllocResponse>> RpcServerCore::Alloc(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                                                           const AllocRequest &request)
{
    UNUSED(sender);
    UNUSED(oplib);

    auto alignment = request.alignment();
    auto num_bytes = request.num_bytes();

    INFO("Serving AllocRequest with alignment {} and num_bytes {}", alignment, num_bytes);

    auto ptr = MemoryMgr::instance().allocate(num_bytes, alignment);
    auto addr_handle = reinterpret_cast<uint64_t>(ptr);

    auto response = std::make_unique<AllocResponse>();
    response->set_addr_handle(addr_handle);

    INFO("Allocated address handel: {:x}", addr_handle);

    response->mutable_result()->set_code(0);
    return ExecutionEngine::instance().makePromise(std::move(response));
}

q::promise<unique_ptr<DeallocResponse>> RpcServerCore::Dealloc(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                                                               const DeallocRequest &request)
{
    UNUSED(sender);
    UNUSED(oplib);

    auto addr_handle = request.addr_handle();

    INFO("Serving DeallocRequest with address handel: {:x}", addr_handle);

    MemoryMgr::instance().deallocate(reinterpret_cast<void*>(addr_handle));

    auto response = std::make_unique<DeallocResponse>();
    response->mutable_result()->set_code(0);
    return ExecutionEngine::instance().makePromise(std::move(response));
}

q::promise<unique_ptr<CustomResponse>> RpcServerCore::Custom(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                                                             const CustomRequest &request)
{
    auto task = oplib->createCustomTask(std::move(sender), request);
    if (!task) {
        ERR("Skipping task due to failed creation.");
        return ExecutionEngine::instance().emptyPromise<CustomResponse>();
    }

    return ExecutionEngine::instance().enqueue<CustomResponse>(std::move(task));
}
