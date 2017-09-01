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

#ifndef LOGGING_H
#define LOGGING_H

#include "spdlog/spdlog.h"

// For additional operator<< implementations to work
#include "spdlog/fmt/ostr.h"

#include <iomanip>
#include <type_traits>

namespace logging {

class LoggerWrapper
{
public:
    LoggerWrapper(std::shared_ptr<spdlog::logger> logger);
    LoggerWrapper(LoggerWrapper &&wrapper);

    spdlog::logger *operator->();

private:
    struct Stream
    {
        explicit Stream(std::shared_ptr<spdlog::logger> &&l)
            : logger(l)
        {
        }
        std::shared_ptr<spdlog::logger> logger;
        ~Stream();
    };
    std::unique_ptr<Stream> m_stream;
};

LoggerWrapper logger();

} // namespace logging

// Additional operator<< implementations
std::ostream &operator<<(std::ostream &os, const std::exception_ptr &ep);

namespace executor {
class OpKernelDef;
class EvenlopDef;
}
std::ostream &operator<<(std::ostream &os, const executor::OpKernelDef &c);
std::ostream &operator<<(std::ostream &os, const executor::EvenlopDef &c);

namespace zmq {
class message_t;
class error_t;
}
std::ostream &operator<<(std::ostream &os, const zmq::message_t &c);
std::ostream &operator<<(std::ostream &os, const zmq::error_t &c);

namespace google {
namespace protobuf {
class Message;
}
}

std::ostream &operator<<(std::ostream &os, const google::protobuf::Message &c);

struct DeviceSpec;
std::ostream &operator<<(std::ostream &os, const DeviceSpec &c);

/**
 * Generic operator<< for strong enum classes
 */
template<typename T>
std::ostream &operator<<(typename std::enable_if<std::is_enum<T>::value, std::ostream>::type &os, const T &e)
{
    return os << static_cast<typename std::underlying_type<T>::type>(e);
}

/**
 * Generic operator<< for pointer types, excluding char*
 */
struct PtrPrintHelper { uint64_t value; };

template<typename T>
PtrPrintHelper as_hex(T *p)
{
    return { reinterpret_cast<uint64_t>(p) };
}

template<typename T>
PtrPrintHelper as_hex(const std::unique_ptr<T> &p)
{
    return { reinterpret_cast<uint64_t>(p.get()) };
}

std::ostream &operator<<(std::ostream &os, const PtrPrintHelper &helper);

#define TRACE(...) logging::logger()->trace(__VA_ARGS__)
#define DEBUG(...) logging::logger()->debug(__VA_ARGS__)
#define INFO(...) logging::logger()->info(__VA_ARGS__)
#define WARN(...) logging::logger()->warn(__VA_ARGS__)
#define ERR(...) logging::logger()->error(__VA_ARGS__)
#define FATAL(...) logging::logger()->critical(__VA_ARGS__)

#endif // LOGGING_H
