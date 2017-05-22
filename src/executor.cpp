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


    void registerOpLibrary(unique_ptr<IOpLibrary> library);

private:
    unordered_map<OpKernelDef::OpLibraryType, unique_ptr<IOpLibrary>> m_opLibraries;
};
} // namespace executor

executor::ExecServiceImpl::ExecServiceImpl()
{
}

void RunServer() {
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

int main(int argc, char** argv) {
    RunServer();

    return 0;
}
