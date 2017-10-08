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

#include "protos.h"

#include <unordered_map>
#include <functional>

using namespace executor;
using ::google::protobuf::Message;
using std::unique_ptr;

q::promise<ProtoPtr> RpcServerCore::dispatch(ZmqServer::Sender sender, const EvenlopDef &evenlop,
                                             const Message &request)
{
    // NOTE: this-> is need to workaround a bug in GCC 6.x where member function lookup is broken
    // for generic lambda. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61636
#define ITEM(name) \
        {"executor." #name "Request", [this](auto &&sender, auto *oplib, \
                                             const auto &evenlop, const auto &request) { \
            return this->name (std::move(sender), oplib, evenlop, static_cast<const name ## Request&>(request)) \
                .then([](unique_ptr<name ## Response> &&f){ \
                    return static_cast<ProtoPtr>(std::move(f)); \
            }); \
        }},

    using ServiceMethod = std::function<q::promise<ProtoPtr>(ZmqServer::Sender &&sender, IOpLibrary*,
                                                             const EvenlopDef&, const Message&)>;
    static std::unordered_map<std::string, ServiceMethod> funcs {
        CALL_ALL_SERVICE_NAME(ITEM)
    };

#undef ITEM

    assert(sender);

    VLOG(1) << "Serving " << evenlop.type() << " for oplibrary " << OpLibraryType_Name(evenlop.oplibrary());

    auto fit = funcs.find(evenlop.type());
    if (fit == funcs.end()) {
        LOG(ERROR) << "Skipping request because requested method not found: " << evenlop.type();
        return ExecutionEngine::instance().emptyPromise<Message>();
    }

    auto oplib = OpLibraryRegistary::instance().findOpLibrary(evenlop.oplibrary());
    if (!oplib) {
        LOG(ERROR) << "Skipping due to failed to find requested OpLibrary.";
        return ExecutionEngine::instance().emptyPromise<Message>();
    }

    return fit->second(std::move(sender), oplib, evenlop, request);
}

q::promise<unique_ptr<RunResponse>> RpcServerCore::Run(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                                                       const EvenlopDef &evenlop,
                                                       const RunRequest &request)
{
    const auto &opdef = request.opkernel();

    VLOG(1) << "Serving RunRequest with opkernel id " << opdef.id();
    assert(oplib->accepts(opdef));

    auto task = oplib->createRunTask(std::move(sender), evenlop, request);
    if (!task) {
        LOG(ERROR) << "Skipping task due to failed creation.";
        return ExecutionEngine::instance().emptyPromise<RunResponse>();
    }

    return ExecutionEngine::instance().enqueue<RunResponse>(std::move(task));
}

q::promise<unique_ptr<RunGraphResponse>> RpcServerCore::RunGraph(ZmqServer::Sender &&sender,
                                                                 IOpLibrary *oplib,
                                                                 const EvenlopDef &evenlop,
                                                                 const RunGraphRequest &request)
{
    VLOG(1) << "Serving RunGraphRequest";

    auto task = oplib->createRunGraphTask(std::move(sender), evenlop, request);
    if (!task) {
        LOG(ERROR) << "Skipping task due to failed creation.";
        return ExecutionEngine::instance().emptyPromise<RunGraphResponse>();
    }

    return ExecutionEngine::instance().enqueue<RunGraphResponse>(std::move(task));
}

q::promise<unique_ptr<AllocResponse>> RpcServerCore::Alloc(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                                                           const EvenlopDef &evenlop,
                                                           const AllocRequest &request)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(oplib);

    auto alignment = request.alignment();
    auto num_bytes = request.num_bytes();

    VLOG(1) << "Serving AllocRequest with alignment " << alignment << " and num_bytes " << num_bytes;

    auto ptr = MemoryMgr::instance().allocate(num_bytes, alignment);
    auto addr_handle = reinterpret_cast<uint64_t>(ptr);

    auto response = std::make_unique<AllocResponse>();
    response->set_addr_handle(addr_handle);

    VLOG(2) << "Allocated address handel: " << as_hex(ptr);

    response->mutable_result()->set_code(0);
    return ExecutionEngine::instance().makePromise(std::move(response));
}

q::promise<unique_ptr<DeallocResponse>> RpcServerCore::Dealloc(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                                                               const EvenlopDef &evenlop,
                                                               const DeallocRequest &request)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(oplib);

    auto addr_handle = request.addr_handle();
    auto ptr = reinterpret_cast<void*>(addr_handle);

    VLOG(1) << "Serving DeallocRequest with address handel: " << as_hex(ptr);

    MemoryMgr::instance().deallocate(ptr);

    auto response = std::make_unique<DeallocResponse>();
    response->mutable_result()->set_code(0);
    return ExecutionEngine::instance().makePromise(std::move(response));
}

q::promise<unique_ptr<CustomResponse>> RpcServerCore::Custom(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                                                             const EvenlopDef &evenlop,
                                                             const CustomRequest &request)
{
    auto task = oplib->createCustomTask(std::move(sender), evenlop, request);
    if (!task) {
        LOG(ERROR) << "Skipping task due to failed creation.";
        return ExecutionEngine::instance().emptyPromise<CustomResponse>();
    }

    return ExecutionEngine::instance().enqueue<CustomResponse>(std::move(task));
}
