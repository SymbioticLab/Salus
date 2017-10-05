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

static auto kListenFlag = "--listen";
static auto kVerboseFlag = "--verbose";
static auto kVModuleFlag = "--vmodule";
static auto kVLogFileFlag = "--vlogfile";

// <program-name> [-v | -vv | -vvv | --verbose=<verbosity>] [--vmodule=<vmodules>] [-l <endpoint>]
static auto USAGE =
R"(Usage:
    <program-name> [options]
    <program-name> --help
    <program-name> --version

AtLast: trAnsparenT deep LeArning Shared execuTion.

Options:
    -h, --help                  Print this help message and exit.
    -V, --version               Print version and exit.
    -l <endpoint>, --listen=<endpoint>
                                ZeroMQ endpoint to listen on.
                                [default: tcp://*:5501]
    -v, --verbose=<verbosity>   Set verbosity level. Valid range: 0-9.
                                [default: 0]
    --vmodule=<vmodules>        Specify verbose level per module.
                                [default: ]
    --vlogfile=<path>           Log file to use for verbose logging.
                                [default: verbose.log]
)"s;

static auto VERSION = R"(<program-name> version 0.1)"s;

auto parseArguments(int argc, char **argv)
{
    string executable(argv[0]);
    auto idx = executable.find_last_of('/');
    if (idx != string::npos) {
        executable = executable.substr(idx + 1);
    }

    regex pattern(R"(<program-name>)");
    USAGE = regex_replace(USAGE, pattern, executable);
    VERSION = regex_replace(VERSION, pattern, executable);
    return docopt::docopt(USAGE,
                          {argv + 1, argv + argc},
                          true,
                          VERSION);
}

void initializeLogging(std::map<std::string, docopt::value> &args)
{
    const auto &verbosity = args[kVerboseFlag].asLong();
    const auto &vmodules = args[kVModuleFlag].asString();
    const auto &logfile = args[kVLogFileFlag].asString();

    using namespace el;
#if !defined(NDEBUG)
    Loggers::addFlag(LoggingFlag::ImmediateFlush);
#endif
    Loggers::addFlag(LoggingFlag::ColoredTerminalOutput);
    Loggers::addFlag(LoggingFlag::AutoSpacing);
    Loggers::addFlag(LoggingFlag::FixedTimeFormat);
    Loggers::addFlag(LoggingFlag::AllowVerboseIfModuleNotSpecified);

    Helpers::installCustomFormatSpecifier(el::CustomFormatSpecifier("%tid", logging::thread_id));

    Configurations conf;
    conf.setToDefault();
    conf.set(Level::Global, ConfigurationType::Format,
             R"([%datetime] [%tid] [%logger] [%levshort] %msg)");
    conf.set(Level::Global, ConfigurationType::SubsecondPrecision, "6");
    conf.set(Level::Global, ConfigurationType::ToFile, "false");

    // Verbose logging goes to file only
    conf.set(Level::Verbose, ConfigurationType::ToFile, "true");
    conf.set(Level::Verbose, ConfigurationType::ToStandardOutput, "false");
    conf.set(Level::Verbose, ConfigurationType::Filename, logfile);

    Loggers::reconfigureAllLoggers(conf);

    Loggers::setVerboseLevel(verbosity);
    Loggers::setVModules(vmodules.c_str());

    // Deprecated spdlog configuration
    constexpr spdlog::level::level_enum vtol[] = {
        spdlog::level::warn,
        spdlog::level::info,
        spdlog::level::debug,
        spdlog::level::trace,
    };
    logging::logger()->set_level(vtol[verbosity > 3 ? 3 : verbosity]);
}

void printConfiguration(std::map<std::string, docopt::value> &args)
{
#if defined(NDEBUG)
    LOG(INFO) << "Running in Release mode";
#else
    LOG(INFO) << "Running in Debug mode";
#endif
    LOG(INFO) << "Verbose logging level:" << el::Loggers::verboseLevel()
              << "file:" << args[kVLogFileFlag].asString();
}

int main(int argc, char **argv)
{
    auto args = parseArguments(argc, argv);

    initializeLogging(args);

    printConfiguration(args);

    LOG(TRACE) << "Trace";
    LOG(DEBUG) << "Debug";
    LOG(INFO) << "Info";
    LOG(WARNING) << "Warning";
    LOG(ERROR) << "Error";
    LOG(FATAL) << "Fatal";
    VLOG(1) << "Vlog 1";

    ZmqServer server(make_unique<RpcServerCore>());

    const auto &listen = args[kListenFlag].asString();
    LOG(INFO) << "Starting server listening at" << listen;
    server.start(listen);

    server.join();

    return 0;
}
