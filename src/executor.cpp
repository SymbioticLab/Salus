#include "executor.grpc.pb.h"

#include "oplibraries/ioplibrary.h"

#include <grpc++/grpc++.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

using namespace std;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::Status;

namespace executor {
class ExecServiceImpl : public IExecEngine::Service
{
public:
    explicit ExecServiceImpl();

    Status run(ServerContext *context, const RunRequest *request, RunResponse *response) override;
    Status allocate(ServerContext *context, const AllocRequest *request,
                    AllocResponse *response) override;
    Status deallocate(ServerContext *context, const DeallocRequest *request,
                      DeallocResponse *response) override;

    void registerOpLibrary(OpKernelDef::OpLibraryType libraryType, unique_ptr<IOpLibrary> library);
    IOpLibrary *findSuitableOpLibrary(const OpKernelDef &opdef);
private:
    unordered_map<OpKernelDef::OpLibraryType, unique_ptr<IOpLibrary>> m_opLibraries;
};
} // namespace executor

executor::ExecServiceImpl::ExecServiceImpl()
{
}

void executor::ExecServiceImpl::registerOpLibrary(OpKernelDef::OpLibraryType libraryType,
                                                  unique_ptr<IOpLibrary> library)
{
    m_opLibraries[libraryType] = std::move(library);
}

IOpLibrary *executor::ExecServiceImpl::findSuitableOpLibrary(const OpKernelDef &opdef)
{
    for (const auto &elem : m_opLibraries) {
        if (elem.first == opdef.oplibrary() && elem.second->accepts(opdef)) {
            return elem.second.get();
        }
    }
}

Status executor::ExecServiceImpl::run(grpc::ServerContext *context,
                                      const executor::RunRequest *request,
                                      executor::RunResponse *response)
{
    auto opdef = request->opkernel();
    auto ctxdef = request->context();

    auto oplib = findSuitableOpLibrary(opdef);
    assert(oplib->accepts(opdef));

    auto task = oplib->createTask(opdef, ctxdef);

    // run the task right away
    auto res = response->mutable_result();
    auto newctxdef = response->mutable_context();

    *res = task->run();
    *newctxdef = task->contextDef();

    // TODO: Enqueue the task, and run async
    return Status::OK;
}

Status executor::ExecServiceImpl::allocate(grpc::ServerContext *context,
                                           const executor::AllocRequest *request,
                                           executor::AllocResponse *response)
{
    auto alignment = request->alignment();
    auto num_bytes = request->num_bytes();

    // TODO: compute addr_handle
    uint64_t addr_handle = 0;
    response->set_addr_handle(addr_handle);

    response->mutable_result()->set_code(0);
    return Status::OK;
}

Status executor::ExecServiceImpl::deallocate(grpc::ServerContext *context,
                                             const executor::DeallocRequest *request,
                                             executor::DeallocResponse *response)
{
    auto addr_handle = request->addr_handle();

    // TODO: do deallocate

    response->mutable_result()->set_code(0);
    return Status::OK;
}

void RunServer()
{
    std::string server_address("0.0.0.0:50051");
    executor::ExecServiceImpl service;

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc, char **argv)
{
    RunServer();

    return 0;
}
