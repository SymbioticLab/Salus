/*
 * Copyright 2019 Peifeng Yu <peifeng@umich.edu>
 * 
 * This file is part of Salus
 * (see https://github.com/SymbioticLab/Salus).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "logging.h"

#include "utils/envutils.h"
#include "utils/stringutils.h"

#include "protos.h"

#include <zmq.hpp>

#include <ostream>

namespace {

class ThreadSafePerformanceTrackingCallback : public el::PerformanceTrackingCallback
{
protected:
    void handle(const el::PerformanceTrackingData *data) override
    {
        using namespace el;
        base::type::stringstream_t ss;
        if (data->dataType() == PerformanceTrackingData::DataType::Complete) {
            ss << "Executed [" << data->blockName()->c_str() << "] in ["
               << *data->formattedTimeTaken() << "]";
        } else {
            ss << "Performance checkpoint";
            if (!data->checkpointId().empty()) {
                ss << " [" << data->checkpointId().c_str() << "]";
            }
            ss << " for block [" << data->blockName()->c_str() << "] : ["
               << *data->performanceTracker() << "]";
        }
        el::base::Writer(data->performanceTracker()->level(), data->file(),data->line(), data->func())
            .construct(1, data->loggerId().c_str())
            << ss.str();
    }
};

} // namespace

INITIALIZE_EASYLOGGINGPP

namespace logging {
void initialize(const Params &params)
{
    using namespace el;

    // WORKAROUND: the default performance tracking callback is not really thread safe.
    // thus before that's fixed, we use our own.
    Helpers::uninstallPerformanceTrackingCallback<el::base::DefaultPerformanceTrackingCallback>(
        "DefaultPerformanceTrackingCallback");
    Helpers::installPerformanceTrackingCallback<ThreadSafePerformanceTrackingCallback>(
        "ThreadSafePerformanceTrackingCallback");

#if !defined(NDEBUG)
    Loggers::addFlag(LoggingFlag::ImmediateFlush);
#endif
    Loggers::addFlag(LoggingFlag::ColoredTerminalOutput);
    Loggers::addFlag(LoggingFlag::FixedTimeFormat);
    Loggers::addFlag(LoggingFlag::AllowVerboseIfModuleNotSpecified);
    Loggers::addFlag(LoggingFlag::DisablePerformanceTrackingCheckpointComparison);

    Helpers::installCustomFormatSpecifier(el::CustomFormatSpecifier("%tid", logging::thread_id));

    Configurations conf;
    conf.setToDefault();
    conf.set(Level::Global, ConfigurationType::Format,
             R"([%datetime{%Y-%M-%d %H:%m:%s.%g}] [%tid] [%logger] [%levshort] %msg)");
    conf.set(Level::Global, ConfigurationType::SubsecondPrecision, "6");
    conf.set(Level::Global, ConfigurationType::ToFile, "false");

    conf.set(Level::Global, ConfigurationType::PerformanceTracking, params.pLogFile ? "true" : "false");

    // Verbose logging goes to file only
    conf.set(Level::Verbose, ConfigurationType::ToFile, "true");
    conf.set(Level::Verbose, ConfigurationType::ToStandardOutput, "false");
    if (params.vLogFile) {
        conf.set(Level::Verbose, ConfigurationType::Filename, *params.vLogFile);
    }

    Loggers::setDefaultConfigurations(conf, true /*configureExistingLoggers*/);

    if (params.verbosity) {
        Loggers::setVerboseLevel(*params.verbosity);
    }
    if (params.vModules) {
        Loggers::setVModules(params.vModules->c_str());
    }
    // Separate configuration for performance logger
    if (params.pLogFile) {
        Configurations perfConf;
        perfConf.set(Level::Info, ConfigurationType::ToFile, "true");
        perfConf.set(Level::Info, ConfigurationType::ToStandardOutput, "false");
        perfConf.set(Level::Info, ConfigurationType::Filename, *params.pLogFile);
        Loggers::reconfigureLogger(logging::kPerfTag, perfConf);
    } else {
        Loggers::reconfigureLogger(logging::kPerfTag, ConfigurationType::Enabled, "false");
    }

    // Force to create loggers here with default configuration
    // in non-performance sensitive code path.
    for (auto tag : {logging::kAllocTag, logging::kOpTracing, logging::kPerfTag, logging::kDefTag}) {
        auto logger = Loggers::getLogger(tag);
        DCHECK(logger);
    }

    // Read in configuration file
    if (params.configFile) {
        Loggers::configureFromGlobal(params.configFile->c_str());
    }
}

} // namespace logging

MAKE_LOGGABLE(std::exception_ptr, ep, os)
{
    try {
        std::rethrow_exception(ep);
    } catch (const std::exception &e) {
        os << e.what();
    } catch (...) {
        os << "unknown exception";
    }
    return os;
}

MAKE_LOGGABLE(executor::OpKernelDef, c, os)
{
    return os << "OpKernelDef(" << c.id() << ", " << executor::OpLibraryType_Name(c.oplibrary()) << ")";
}

MAKE_LOGGABLE(executor::EvenlopDef, c, os)
{
    return os << "EvenlopDef(type='" << c.type() << "', seq=" << c.seq() << ", sess=" << c.sessionid()
              << ", recvId='"
              << sstl::bytesToHexString(reinterpret_cast<const uint8_t *>(c.recvidentity().data()),
                                         c.recvidentity().size())
              << "')";
}

MAKE_LOGGABLE(zmq::error_t, c, os)
{
    return os << "zmq::error_t(code=" << c.num() << ", msg='" << c.what() << "')";
}

MAKE_LOGGABLE(google::protobuf::Message, c, os)
{
    return os << c.DebugString();
}

MAKE_LOGGABLE(PtrPrintHelper, helper, os)
{
    std::ios state(nullptr);
    state.copyfmt(os);
    os << std::showbase << std::hex << helper.value;
    os.copyfmt(state);
    return os;
}
