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

namespace logging {

class LoggerWrapper
{
public:
    LoggerWrapper();
    ~LoggerWrapper();

    static std::shared_ptr<spdlog::logger> &logger();

private:
    std::shared_ptr<spdlog::logger> m_logger;
};

} // namespace logging

// Additional operator<< implementations
namespace executor {
class OpKernelDef;
}
std::ostream &operator<<(std::ostream &os, const executor::OpKernelDef &c);

namespace zmq {
class message_t;
}
std::ostream &operator<<(std::ostream &os, const zmq::message_t &c);

#define TRACE(...) logging::LoggerWrapper::logger()->trace(__VA_ARGS__)
#define DEBUG(...) logging::LoggerWrapper::logger()->debug(__VA_ARGS__)
#define INFO(...) logging::LoggerWrapper::logger()->info(__VA_ARGS__)
#define WARN(...) logging::LoggerWrapper::logger()->warn(__VA_ARGS__)
#define ERR(...) logging::LoggerWrapper::logger()->error(__VA_ARGS__)
#define FATAL(...) logging::LoggerWrapper::logger()->critical(__VA_ARGS__)

#endif // LOGGING_H
