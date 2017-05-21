#include "executor.grpc.pb.h"

#include "ioplibrary.h"

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
    unordered_map<OpKernel::OpLibraryType, unique_ptr<IOpLibrary>> m_opLibraries;
};
} // namespace executor

executor::ExecServiceImpl::ExecServiceImpl()
{
}

