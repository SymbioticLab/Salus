#include "oplibraries/ioplibrary.h"
#include "rpcserver/zmqserver.h"
#include "rpcserver/rpcservercore.h"
#include "platform/logging.h"
#include "utils/cpp17.h"
#include "utils/macros.h"

#include "docopt.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <regex>

using namespace std;
using namespace std::string_literals;

namespace {
static auto kListenFlag = "--listen";
static auto kLogConfFlag = "--logconf";
static auto kVerboseFlag = "--verbose";
static auto kVModuleFlag = "--vmodule";
static auto kVLogFileFlag = "--vlogfile";
static auto kPLogFileFlag = "--perflog";

// <program-name> [-v | -vv | -vvv | --verbose=<verbosity>] [--vmodule=<vmodules>] [-l <endpoint>]
static auto kUsage =
R"(Usage:
    <program-name> [options]
    <program-name> --help
    <program-name> --version

AtLast: trAnsparenT deep LeArning Shared execuTion.

Options:
    -h, --help                  Print this help message and exit.
    -V, --version               Print version and exit.
    -l <endpoint>, --listen=<endpoint>
                                Listen on ZeroMQ endpoint <endpoint>.
                                [default: tcp://*:5501]
    --logconf <file>            Path to log configuration file. Note that
                                settings in this file takes precedence over
                                other command line arguments.)"
#if !defined(ELPP_DISABLE_VERBOSE_LOGS)
R"(
    -v <level>, --verbose=<level>
                                Enable verbose logging level <level>.
                                Valid range: 0-9. (0 means disable)
                                [default: 0]
    --vmodule=<vmodules>        Specify verbose level per module.
                                Refer to https://github.com/muflihun/easyloggingpp#vmodule
                                for syntax.
                                [default: ]
    --vlogfile=<file>           Verbose logging goes to <file>.
                                [default: verbose.log])"
#endif
#if !defined(ELPP_DISABLE_PERFORMANCE_TRACKING)
R"(
    --perflog=<file>            Enable performance logging and log to <file>.
)"
#endif
""s;

static auto kVersion = R"(AtLast: trAnsparenT deep LeArning Shared execuTion version 0.1)"s;

template<typename T>
utils::optional<T> optional_arg(const docopt::value &v);

template<>
inline utils::optional<std::string> optional_arg(const docopt::value &v)
{
    if (v) return v.asString();
    return utils::nullopt;
}

template<>
inline utils::optional<int> optional_arg(const docopt::value &v)
{
    if (v) return v.asLong();
    return utils::nullopt;
}

template<>
inline utils::optional<bool> optional_arg(const docopt::value &v)
{
    if (v) return v.asBool();
    return utils::nullopt;
}

} // namespace

auto parseArguments(int argc, char **argv)
{
    string executable(argv[0]);
    auto idx = executable.find_last_of('/');
    if (idx != string::npos) {
        executable = executable.substr(idx + 1);
    }

    regex pattern(R"(<program-name>)");
    kUsage = regex_replace(kUsage, pattern, executable);
    kVersion = regex_replace(kVersion, pattern, executable);
    return docopt::docopt(kUsage,
                          {argv + 1, argv + argc},
                          true,
                          kVersion);
}

void initializeLogging(std::map<std::string, docopt::value> &args)
{
    logging::initialize({
        optional_arg<std::string>(args[kLogConfFlag]),
        optional_arg<int>(args[kVerboseFlag]),
        optional_arg<std::string>(args[kVModuleFlag]),
        optional_arg<std::string>(args[kVLogFileFlag]),
        optional_arg<std::string>(args[kPLogFileFlag]),
    });
}

void printConfiguration(std::map<std::string, docopt::value> &)
{
#if defined(NDEBUG)
    LOG(INFO) << "Running in Release mode";
#else
    LOG(INFO) << "Running in Debug mode";
#endif
    {
        const auto &conf = el::Loggers::getLogger(logging::kDefTag)->typedConfigurations();
        LOG(INFO) << "Verbose logging level: " << el::Loggers::verboseLevel()
                  << " file: " << conf->filename(el::Level::Verbose);
    }
    {
        const auto &conf = el::Loggers::getLogger(logging::kPerfTag)->typedConfigurations();
        LOG(INFO) << "Performance logging: " << (conf->enabled(el::Level::Info) ? "enabled" : "disabled")
                  << " file: " << conf->filename(el::Level::Info);
    }
    {
        const auto &conf = el::Loggers::getLogger(logging::kAllocTag)->typedConfigurations();
        LOG(INFO) << "Allocation logging: " << (conf->enabled(el::Level::Info) ? "enabled" : "disabled");
    }
}

int main(int argc, char **argv)
{
    auto args = parseArguments(argc, argv);

    initializeLogging(args);

    printConfiguration(args);

    AllocLog(INFO) << "Test log using alloc logger";

    ZmqServer server(make_unique<RpcServerCore>());

    const auto &listen = args[kListenFlag].asString();
    LOG(INFO) << "Starting server listening at " << listen;
    server.start(listen);

    server.join();

    return 0;
}
