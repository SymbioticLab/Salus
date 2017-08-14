#include "oplibraries/ioplibrary.h"
#include "rpcserver/zmqserver.h"
#include "rpcserver/rpcservercore.h"
#include "platform/logging.h"
#include "utils/macros.h"

#include "docopt.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <regex>

using namespace std;
using namespace std::string_literals;

static auto USAGE =
R"(Usage:
    <program-name> [-vvv] [-l <endpoint>]
    <program-name> --help
    <program-name> --version

Background execution service for deep learning frameworks.

Options:
    -h, --help                  Print this help message and exit.
    -V, --version               Print version and exit.
    -l <endpoint>, --listen=<endpoint>
                                ZeroMQ endpoint to listen on.
                                [default: tcp://*:5501]
    -v, --verbose               Verbose. Can be specified up to three times for
                                more verbosity.
)"s;

static auto VERSION = R"(<program-name> version 0.1)"s;

int main(int argc, char **argv)
{
    string executable(argv[0]);
    auto idx = executable.find_last_of('/');
    if (idx != string::npos) {
        executable = executable.substr(idx + 1);
    }

    regex pattern(R"(<program-name>)");
    USAGE = regex_replace(USAGE, pattern, executable);
    VERSION = regex_replace(VERSION, pattern, executable);
    auto args = docopt::docopt(USAGE,
                               {argv + 1, argv + argc},
                               true,
                               VERSION);

    const auto &listen = args["--listen"].asString();
    const auto &verbose = args["--verbose"].asLong();

    constexpr spdlog::level::level_enum vtol[] = {
        spdlog::level::warn,
        spdlog::level::info,
        spdlog::level::debug,
        spdlog::level::trace,
    };
    logging::logger()->set_level(vtol[verbose]);

    ZmqServer server(make_unique<RpcServerCore>());

    INFO("Starting server listening at {}", listen);
    server.start(listen);

    server.join();

    return 0;
}
