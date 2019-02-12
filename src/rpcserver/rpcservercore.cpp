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

#include "rpcservercore.h"

#include "execution/executionengine.h"
#include "resources/memorymgr.h"
#include "oplibraries/ioplibrary.h"
#include "platform/logging.h"
#include "utils/macros.h"

#include "protos.h"

#include <functional>
#include <unordered_map>

using namespace executor;
using ::google::protobuf::Message;
using std::unique_ptr;

RpcServerCore::RpcServerCore()
{
    OpLibraryRegistary::instance().initializeLibraries();
}

RpcServerCore::~RpcServerCore()
{
    OpLibraryRegistary::instance().uninitializeLibraries();
}

void RpcServerCore::dispatch(ZmqServer::Sender sender, const EvenlopDef &evenlop, const Message &request)
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
        return;
    }

    auto oplib = OpLibraryRegistary::instance().findOpLibrary(evenlop.oplibrary());
    if (!oplib) {
        LOG(ERROR) << "Skipping due to failed to find requested OpLibrary.";
        return;
    }

    fit->second(std::move(sender), oplib, evenlop, request);
}

void RpcServerCore::Run(ZmqServer::Sender &&sender, IOpLibrary *oplib, const EvenlopDef &evenlop,
                        const RunRequest &request)
{
    const auto &opdef = request.opkernel();

    VLOG(2) << "Serving RunRequest with opkernel id " << opdef.id();
    DCHECK(oplib->accepts(opdef));

    oplib->onRun(sender, evenlop, request, [sender](auto resp) {
        if (resp) {
            sender->sendMessage(std::move(resp));
        }
    });
}

void RpcServerCore::RunGraph(ZmqServer::Sender &&sender, IOpLibrary *oplib, const EvenlopDef &evenlop,
                             const RunGraphRequest &request)
{
    VLOG(2) << "Serving RunGraphRequest";

    oplib->onRunGraph(sender, evenlop, request, [sender](auto resp) {
        if (resp) {
            sender->sendMessage(std::move(resp));
        }
    });
}

void RpcServerCore::Alloc(ZmqServer::Sender &&sender, IOpLibrary *oplib, const EvenlopDef &evenlop,
                          const AllocRequest &request)
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

void RpcServerCore::Dealloc(ZmqServer::Sender &&sender, IOpLibrary *oplib, const EvenlopDef &evenlop,
                            const DeallocRequest &request)
{
    UNUSED(sender);
    UNUSED(evenlop);
    UNUSED(oplib);

    auto addr_handle = request.addr_handle();
    auto ptr = reinterpret_cast<void *>(addr_handle);

    VLOG(2) << "Serving DeallocRequest with address handel: " << as_hex(ptr);

    MemoryMgr::instance().deallocate(ptr);

    auto response = std::make_unique<DeallocResponse>();
    response->mutable_result()->set_code(0);
    sender->sendMessage(std::move(response));
}

void RpcServerCore::Custom(ZmqServer::Sender &&sender, IOpLibrary *oplib, const EvenlopDef &evenlop,
                           const CustomRequest &request)
{
    oplib->onCustom(sender, evenlop, request, [sender](auto resp) {
        if (resp) {
            sender->sendMessage(std::move(resp));
        }
    });
}
