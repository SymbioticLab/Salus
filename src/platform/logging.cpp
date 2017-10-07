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
#include "utils/envutils.h"
#include "utils/stringutils.h"

#include "protos.h"

#include <zmq.hpp>

namespace {

uint64_t maxBytesDumpLen()
{
    return utils::fromEnvVarCached("EXEC_MAX_BYTES_DUMP_LEN", UINT64_C(20));
}

struct LoggerStaticInitializer
{
    std::shared_ptr<spdlog::logger> logger;
    LoggerStaticInitializer()
    {
#ifdef NDEBUG
        spdlog::set_async_mode(8192);
#endif
        logger = spdlog::stdout_color_mt("console");
        logger->set_pattern("[%Y-%m-%d %T.%F] [%t] [%n] [%L] %v");

#ifdef NDEBUG
        logger->flush_on(spdlog::level::err);
        logger->set_level(spdlog::level::err);
#else
        logger->flush_on(spdlog::level::trace);
        logger->set_level(spdlog::level::trace);
#endif
        //         g3::installCrashHandler();
        //         g3::setDumpStack(false);
    }
};

} // namespace

INITIALIZE_EASYLOGGINGPP

logging::LoggerWrapper logging::logger()
{
    static LoggerStaticInitializer init;

    return LoggerWrapper(init.logger);
}

logging::LoggerWrapper::LoggerWrapper(std::shared_ptr<spdlog::logger> logger)
    : m_stream(std::make_unique<Stream>(std::move(logger)))
{
}

logging::LoggerWrapper::LoggerWrapper(LoggerWrapper &&wrapper)
    : m_stream(std::move(wrapper.m_stream))
{
}

logging::LoggerWrapper::Stream::~Stream() = default;

spdlog::logger *logging::LoggerWrapper::operator->()
{
    if (m_stream)
        return m_stream->logger.get();
    return nullptr;
}

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
              << utils::bytesToHexString(reinterpret_cast<const uint8_t *>(c.recvidentity().data()),
                                         c.recvidentity().size())
              << "')";
}

MAKE_LOGGABLE(zmq::message_t, c, os)
{
    return os << "zmq::message_t(len=" << c.size() << ", data='"
              << utils::bytesToHexString(c.data<uint8_t>(), c.size(), maxBytesDumpLen()) << "')";
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
    return os << std::showbase << std::hex << helper.value;
}
