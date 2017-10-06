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
static auto kPLogFileFlag = "--perflog";

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
                                Listen on ZeroMQ endpoint <endpoint>.
                                [default: tcp://*:5501]
    -v <level>, --verbose=<level>
                                Enable verbose logging level <level>.
                                Valid range: 0-9. (0 means disable)
                                [default: 0]
    --vmodule=<vmodules>        Specify verbose level per module.
                                Refer to https://github.com/muflihun/easyloggingpp#vmodule
                                [default: ]
    --vlogfile=<file>           Verbose logging goes to <file>
                                [default: verbose.log]
    --perflog=<file>            Enable performance logging and log to <file>.
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
    const auto &perflog = args[kPLogFileFlag];

    using namespace el;
#if !defined(NDEBUG)
    Loggers::addFlag(LoggingFlag::ImmediateFlush);
#endif
    Loggers::addFlag(LoggingFlag::ColoredTerminalOutput);
    Loggers::addFlag(LoggingFlag::FixedTimeFormat);
    Loggers::addFlag(LoggingFlag::AllowVerboseIfModuleNotSpecified);

    Helpers::installCustomFormatSpecifier(el::CustomFormatSpecifier("%tid", logging::thread_id));

    Configurations conf;
    conf.setToDefault();
    conf.set(Level::Global, ConfigurationType::Format,
             R"([%datetime{%Y-%M-%d %H:%m:%s.%g}] [%tid] [%logger] [%levshort] %msg)");
    conf.set(Level::Global, ConfigurationType::SubsecondPrecision, "6");
    conf.set(Level::Global, ConfigurationType::ToFile, "false");
    if (perflog) {
        conf.set(Level::Global, ConfigurationType::PerformanceTracking, "true");
    } else {
        conf.set(Level::Global, ConfigurationType::PerformanceTracking, "false");
    }

    // Verbose logging goes to file only
    conf.set(Level::Verbose, ConfigurationType::ToFile, "true");
    conf.set(Level::Verbose, ConfigurationType::ToStandardOutput, "false");
    conf.set(Level::Verbose, ConfigurationType::Filename, logfile);

    Loggers::setDefaultConfigurations(conf, true /*configureExistingLoggers*/);

    Loggers::setVerboseLevel(verbosity);
    Loggers::setVModules(vmodules.c_str());

    // Separate configuration for performance logger
    if (perflog) {
        Configurations perfConf;
        perfConf.set(Level::Info, ConfigurationType::ToFile, "true");
        perfConf.set(Level::Info, ConfigurationType::ToStandardOutput, "false");
        perfConf.set(Level::Info, ConfigurationType::Filename, perflog.asString());
        Loggers::reconfigureLogger("performance", perfConf);
    }

    // Separate allocation logger, which uses default configuration. Force to create it here
    // in non-performance sensitive code path.
    auto allocLogger = Loggers::getLogger("alloc");
    DCHECK(allocLogger);

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
    LOG(INFO) << "Verbose logging level: " << el::Loggers::verboseLevel()
              << " file:" << args[kVLogFileFlag].asString();

    const auto &conf = el::Loggers::getLogger("performance")->typedConfigurations();
    LOG(INFO) << "Performance logging: " << (conf->enabled(el::Level::Info) ? "enabled" : "disabled")
              << " file: " << conf->filename(el::Level::Info);
}

int main(int argc, char **argv)
{
    auto args = parseArguments(argc, argv);

    initializeLogging(args);

    printConfiguration(args);

    ZmqServer server(make_unique<RpcServerCore>());

    const auto &listen = args[kListenFlag].asString();
    LOG(INFO) << "Starting server listening at " << listen;
    server.start(listen);

    server.join();

    return 0;
}
