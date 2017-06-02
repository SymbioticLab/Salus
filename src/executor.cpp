#include "oplibraries/ioplibrary.h"
#include "rpcserver/zmqserver.h"
#include "rpcserver/rpcservercore.h"
#include "platform/logging.h"
#include "utils/macros.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

using namespace std;

int main(int argc, char **argv)
{
    UNUSED(argc);
    UNUSED(argv);

    ZmqServer server;
    server.setNumWorkers(1);

    INFO("Starting server");
    server.start(make_unique<RpcServerCore>(), "tcp://*:5501", false);
    server.join();
    server.stop();

    return 0;
}
