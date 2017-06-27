/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017  Aetf <aetf@unlimitedcodeworks.xyz>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "logging.h"

#include "crashhandler/crashhandler.hpp"
#include "utils/stringutils.h"
#include "execution/devices.h"

#include "executor.pb.h"

#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/lib/core/status.h>

#include <google/protobuf/message.h>

#include <zmq.hpp>

namespace {
// Parse log level (int64) from environment variable (char*)
uint64_t logLevelStrToIntOr(const char* env_var_val, uint64_t val) {
    if (env_var_val == nullptr) {
        return val;
    }

    // Ideally we would use env_var / safe_strto64, but it is
    // hard to use here without pulling in a lot of dependencies,
    // so we use std:istringstream instead
    std::string levelstr(env_var_val);
    std::istringstream ss(levelstr);
    uint64_t level;
    if (!(ss >> level)) {
        // Invalid vlog level setting, set level to default (0)
        level = val;
    }

    return level;
}

uint64_t maxBytesDumpLenFromEnv()
{
    const char* env_var_val = std::getenv("EXEC_MAX_BYTES_DUMP_LEN");
    return logLevelStrToIntOr(env_var_val, 20);
}

uint64_t maxBytesDumpLen()
{
    static uint64_t len = maxBytesDumpLenFromEnv();
    return len;
}

} // namespace

logging::LoggerWrapper::LoggerWrapper()
{
    spdlog::set_async_mode(8192);
    m_logger = spdlog::stdout_color_mt("console");

    m_logger->flush_on(spdlog::level::trace);
    m_logger->set_level(spdlog::level::trace);
    m_logger->set_pattern("[%Y-%m-%d %T.%e] [%t] [%n] [%L] %v");

    //g3::installCrashHandler();
}

logging::LoggerWrapper::~LoggerWrapper() = default;

std::shared_ptr<spdlog::logger> &logging::LoggerWrapper::logger()
{
    static LoggerWrapper wrapper;
    return wrapper.m_logger;
}

std::ostream &operator<<(std::ostream &os, const std::exception_ptr &ep)
{
    try {
        std::rethrow_exception(ep);
    } catch (const std::exception& e) {
        os << e.what();
    } catch (...) {
        os << "unknown exception";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const executor::OpKernelDef &c)
{
    return os << "OpKernelDef(" << c.id() << ", "
              << executor::OpLibraryType_Name(c.oplibrary()) << ")";
}

std::ostream &operator<<(std::ostream &os, const executor::EvenlopDef &c)
{
    return os << "EvenlopDef(type='" << c.type()
              << "', seq=" << c.seq()
              << ", recvId='"
              << utils::bytesToHexString(reinterpret_cast<const uint8_t*>(c.recvidentity().data()),
                                         c.recvidentity().size())
              << "')";
}

std::ostream &operator<<(std::ostream &os, const zmq::message_t &c)
{
    return os << "zmq::message_t(len=" << c.size()
              << ", data='"
              << utils::bytesToHexString(c.data<uint8_t>(), c.size(), maxBytesDumpLen()) << "')";
}

std::ostream &operator<<(std::ostream &os, const zmq::error_t &c)
{
    return os << "zmq::error_t(code=" << c.num()
              << ", msg='" << c.what() << "')";
}

std::ostream &operator<<(std::ostream &os, const tensorflow::AllocatorAttributes &c)
{
    return os << "tensorflow::AllocatorAttributes("
    << "on_host=" << c.on_host()
    << ", nic_compatible=" << c.nic_compatible()
    << ", gpu_compatible=" << c.gpu_compatible()
    << ", track_sizes=" << c.track_sizes()
    << ")";
}

std::ostream &operator<<(std::ostream &os, const google::protobuf::Message &c)
{
    return os << c.DebugString();
}

std::ostream &operator<<(std::ostream &os, const DeviceSpec &c)
{
    return os << "DeviceSpec(type=" << c.type << ", id=" << c.id << ")";
}
