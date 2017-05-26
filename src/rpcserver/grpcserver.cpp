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

#include "grpcserver.h"

#include "rpcservercore.h"

#include <grpc++/grpc++.h>

using namespace executor;
using grpc::Status;
using grpc::ServerContext;
using grpc::ServerBuilder;
using grpc::Server;

GRpcServer::GRpcServer() { }

GRpcServer::~GRpcServer() = default;

void GRpcServer::start(std::unique_ptr<RpcServerCore> &&logic, const std::string &address, bool block)
{
    // Take ownership of the logic
    m_pLogic = std::move(logic);

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(this);
    // Finally assemble the server.
    m_server = builder.BuildAndStart();
    std::cout << "Server listening on " << address << std::endl;

    if (block) {
        // Wait for the server to shutdown. Note that some other thread must be
        // responsible for shutting down the server for this call to ever return.
        m_server->Wait();
    }
}

void GRpcServer::join()
{
    if (m_server) {
        m_server->Wait();
    }
}

void GRpcServer::stop()
{
    if (m_server) {
        m_server->Shutdown();
        m_server.reset();
    }
}

Status GRpcServer::run(ServerContext *context, const RunRequest *request, RunResponse *response)
{
    m_pLogic->Run(request, response);
    return Status::OK;
}

Status GRpcServer::allocate(grpc::ServerContext *context, const AllocRequest *request,
                            AllocResponse *response)
{
    m_pLogic->Alloc(request, response);
    return Status::OK;
}

Status GRpcServer::deallocate(grpc::ServerContext *context, const DeallocRequest *request,
                              DeallocResponse *response)
{
    m_pLogic->Dealloc(request, response);
    return Status::OK;
}
