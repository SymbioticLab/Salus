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

namespace {
template<typename ResponseType>
void processWithResponse(PTask &&task, ZmqServer::Sender sender)
{
    DCHECK(task);
    DCHECK(sender);
    ExecutionEngine::instance().pool().run([task = std::move(task), sender = std::move(sender)]() {
        auto pResponse = task->run<ResponseType>();
        if (pResponse) {
            sender->sendMessage(std::move(pResponse));
        }
    });
}
} // namespace

RpcServerCore::RpcServerCore()
{
    OpLibraryRegistary::instance().initializeLibraries();
}

RpcServerCore::~RpcServerCore()
{
    OpLibraryRegistary::instance().uninitializeLibraries();
}

void RpcServerCore::dispatch(ZmqServer::Sender sender, const EvenlopDef &evenlop,
                             const Message &request)
{
    // NOTE: this-> is need to workaround a bug in GCC 6.x where member function lookup is broken
    // for generic lambda. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61636
#define ITEM(name) \
        {"executor." #name "Request", [this](auto &&sender, auto *oplib, \
                                             const auto &evenlop, const auto &request) { \
            return this->name (std::move(sender), oplib, evenlop, static_cast<const name ## Request&>(request)); \
        }},

    using ServiceMethod = std::function<void(ZmqServer::Sender &&sender, IOpLibrary*,
                                             const EvenlopDef&, const Message&)>;
    static std::unordered_map<std::string, ServiceMethod> funcs {
        CALL_ALL_SERVICE_NAME(ITEM)
    };

#undef ITEM

    DCHECK(sender);

    VLOG(2) << "Serving " << evenlop.type() << " for oplibrary " << OpLibraryType_Name(evenlop.oplibrary());

    auto fit = funcs.find(evenlop.type());
    if (fit == funcs.end()) {
        LOG(ERROR) << "Skipping request because requested method not found: " << evenlop.type();
    }

    auto oplib = OpLibraryRegistary::instance().findOpLibrary(evenlop.oplibrary());
    if (!oplib) {
        LOG(ERROR) << "Skipping due to failed to find requested OpLibrary.";
    }

    fit->second(std::move(sender), oplib, evenlop, request);
}

void RpcServerCore::Run(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                        const EvenlopDef &evenlop, const RunRequest &request)
{
    const auto &opdef = request.opkernel();

    VLOG(2) << "Serving RunRequest with opkernel id " << opdef.id();
    assert(oplib->accepts(opdef));

    auto task = oplib->createRunTask(sender, evenlop, request);
    if (!task) {
        LOG(ERROR) << "Skipping task due to failed creation.";
    }

    processWithResponse<RunResponse>(std::move(task), std::move(sender));
}

void RpcServerCore::RunGraph(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                             const EvenlopDef &evenlop, const RunGraphRequest &request)
{
    VLOG(2) << "Serving RunGraphRequest";

    auto task = oplib->createRunGraphTask(sender, evenlop, request);
    if (!task) {
        LOG(ERROR) << "Skipping task due to failed creation.";
    }

    processWithResponse<RunResponse>(std::move(task), std::move(sender));
}

void RpcServerCore::Alloc(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                          const EvenlopDef &evenlop, const AllocRequest &request)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(oplib);

    auto alignment = request.alignment();
    auto num_bytes = request.num_bytes();

    VLOG(2) << "Serving AllocRequest with alignment " << alignment << " and num_bytes " << num_bytes;

    auto ptr = MemoryMgr::instance().allocate(num_bytes, alignment);
    auto addr_handle = reinterpret_cast<uint64_t>(ptr);

    auto response = std::make_unique<AllocResponse>();
    response->set_addr_handle(addr_handle);

    VLOG(2) << "Allocated address handel: " << as_hex(ptr);

    response->mutable_result()->set_code(0);
    sender->sendMessage(std::move(response));
}

void RpcServerCore::Dealloc(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                            const EvenlopDef &evenlop, const DeallocRequest &request)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(oplib);

    auto addr_handle = request.addr_handle();
    auto ptr = reinterpret_cast<void*>(addr_handle);

    VLOG(2) << "Serving DeallocRequest with address handel: " << as_hex(ptr);

    MemoryMgr::instance().deallocate(ptr);

    auto response = std::make_unique<DeallocResponse>();
    response->mutable_result()->set_code(0);
    sender->sendMessage(std::move(response));
}

void RpcServerCore::Custom(ZmqServer::Sender &&sender, IOpLibrary *oplib,
                           const EvenlopDef &evenlop, const CustomRequest &request)
{
    auto task = oplib->createCustomTask(sender, evenlop, request);
    if (!task) {
        LOG(ERROR) << "Skipping task due to failed creation.";
    }

    processWithResponse<CustomResponse>(std::move(task), std::move(sender));
}
