#include "execution/resources.h"
#include "execution/executionengine.h"
#include "rpcserver/zmqserver.h"
#include "platform/logging.h"
#include "utils/cpp17.h"
#include "utils/macros.h"

#include <docopt.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <regex>

using namespace std;
using namespace std::string_literals;

namespace {
static auto kListenFlag = "--listen";
static auto kMaxHolWaiting = "--max-hol-waiting";
static auto kDisableFairness = "--disable-fairness";
static auto kDisableAdmissionControl = "--disable-adc";
static auto kDisableWorkConservative = "--disable-wc";
static auto kScheduler = "--sched";

static auto kRandomizedExecution = "--random-exec";

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

Salus: Fine-Grained GPU Sharing for DNN.

Options:
    -h, --help                  Print this help message and exit.
    -V, --version               Print version and exit.
    -l <endpoint>, --listen=<endpoint>
                                Listen on ZeroMQ endpoint <endpoint>.
                                [default: tcp://*:5501]
    --sched                     Scheduler to use. Choices: fair, preempt, pack.
                                [default: fair]
    --disable-adc               Disable admission control.
    --disable-fairness          Disable fair sharing in scheduling.
    --disable-wc                Disable work conservation. Only have effect when
                                fairness is on.
    --random-exec               Using randomized execution for tasks.
    --max-hol-waiting=<num>     Maximum number of task allowed go before queue head
                                in scheduling. [default: 50]
    --logconf <file>            Path to log configuration file. Note that
                                settings in this file takes precedence over
                                other command line arguments.
    -v <level>, --verbose=<level>
                                Enable verbose logging level <level>.
                                Valid range: 0-9. (0 means disable)
                                [default: 0]
    --vmodule=<vmodules>        Specify verbose level per module.
                                Refer to https://github.com/muflihun/easyloggingpp#vmodule
                                for syntax.
                                [default: ]
    --vlogfile=<file>           Verbose logging goes to <file>.
                                [default: verbose.log]
    --perflog=<file>            Enable performance logging and log to <file>.
)"s;

static auto kVersion = R"(Salus: Fine-Grained GPU Sharing for DNN version 0.1)"s;

template<typename T, typename R>
class value_or_helper
{
    using docopt_long_t = typename std::result_of<decltype(&docopt::value::asLong)(docopt::value)>::type;

    static constexpr bool is_string = std::is_same<T, std::string>::value;
    static constexpr bool is_bool = std::is_same<T, bool>::value;
    static constexpr bool is_long = std::is_same<T, long>::value
                                    || (std::is_integral<T>::value
                                        && !is_bool
                                        && sizeof(T) <= sizeof(docopt_long_t));

    static_assert(is_string || is_bool || is_long, "docopt::value only supports std::string, bool and long");

    struct string_tag {};
    struct bool_tag {};
    struct long_tag {};
    struct dispatcher {
    private:
        using bool_or_string = typename std::conditional<is_bool, bool_tag, string_tag>::type;
    public:
        using type = typename std::conditional<is_long, long_tag, bool_or_string>::type;
    };

    value_or_helper(const docopt::value &v, const R &def, string_tag)
        : value(v ? v.asString() : def) { }

    value_or_helper(const docopt::value &v, const R &def, bool_tag)
        : value(v ? v.asBool() : def) { }

    value_or_helper(const docopt::value &v, const R &def, long_tag)
        : value(v ? v.asLong() : def) { }

public:
    value_or_helper(const docopt::value &v, const R &def)
        : value_or_helper(v, def, typename dispatcher::type {}) {}

    typename std::enable_if_t<is_string || is_bool || is_long, R> value;
};

template<typename T, typename R>
inline R value_or(const docopt::value &v, const R &def)
{
    return value_or_helper<T, R>(v, def).value;
}

template<typename T>
inline utils::optional<T> optional_arg(const docopt::value &v)
{
    return value_or<T, utils::optional<T>>(v, utils::nullopt);
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

void configureExecution(std::map<std::string, docopt::value> &args)
{
    auto disableAdmissionControl = value_or<bool>(args[kDisableAdmissionControl], false);
    SessionResourceTracker::instance().setDisabled(disableAdmissionControl);

    auto disableFairness = value_or<bool>(args[kDisableFairness], false);
    uint64_t maxQueueHeadWaiting = value_or<long>(args[kMaxHolWaiting], 50);
    auto randomizedExecution = value_or<bool>(args[kRandomizedExecution], false);
    auto disableWorkConservative = value_or<bool>(args[kDisableWorkConservative], false);
    auto sched = value_or<std::string>(args[kScheduler], "fair"s);

    // Handle deprecated arguments
    if (disableFairness) {
        sched = "pack";
    }

    ExecutionEngine::instance().setSchedulingParam({
        !disableFairness, /* useFairnessCounter */
        maxQueueHeadWaiting,
        randomizedExecution,
        !disableWorkConservative,
        sched
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
    LOG(INFO) << "Admission control: " << (SessionResourceTracker::instance().disabled() ? "off" : "on");
    LOG(INFO) << "Scheduling parameters:";
    auto &param = ExecutionEngine::instance().schedulingParam();
    LOG(INFO) << "    Policy: " << param.scheduler;
    LOG(INFO) << "    MaxQueueHeadWaiting: " << param.maxHolWaiting;
    LOG(INFO) << "    RandomizedExecution: " << (param.randomizedExecution ? "on" : "off");
    LOG(INFO) << "    WorkConservative: " << (param.workConservative ? "on" : "off");
}

int main(int argc, char **argv)
{
    auto args = parseArguments(argc, argv);

    initializeLogging(args);

    configureExecution(args);

    printConfiguration(args);

    auto &server = ZmqServer::instance();

    const auto &listen = args[kListenFlag].asString();
    LOG(INFO) << "Starting server listening at " << listen;
    server.start(listen);

    server.join();

    return 0;
}
