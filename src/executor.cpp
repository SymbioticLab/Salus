#include "oplibraries/ioplibrary.h"
#include "rpcserver/zmqserver.h"
#include "rpcserver/rpcservercore.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

using namespace std;


int main(int argc, char **argv)
{
    ZmqServer server;
    server.setNumWorkers(1);

    cout << "Starting server" << endl;
    server.start(make_unique<RpcServerCore>(), "tcp://127.0.0.1:5501", false);
    server.join();
    server.stop();

    return 0;
}
