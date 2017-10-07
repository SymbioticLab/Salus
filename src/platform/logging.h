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

#include "easylogging++.h"

#include "spdlog/spdlog.h"

// For additional operator<< implementations to work
#include "spdlog/fmt/ostr.h"

#include <iomanip>
#include <type_traits>

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX // prevent windows redefining min/max
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>      // _get_osfhandle and _isatty support
#include <process.h> //  _get_pid support
#include <windows.h>

#ifdef __MINGW32__
#include <share.h>
#endif

#else // unix

#include <fcntl.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/syscall.h> //Use gettid() syscall under linux to get thread id

#elif __FreeBSD__
#include <sys/thr.h> //Use thr_self() syscall under FreeBSD to get thread id
#endif

#endif // unix

#ifndef __has_feature      // Clang - feature checking macros.
#define __has_feature(x) 0 // Compatibility with non-clang compilers.
#endif

namespace logging {

class LoggerWrapper
{
public:
    explicit LoggerWrapper(std::shared_ptr<spdlog::logger> logger);
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

/*
 * Return current thread id as size_t
 * It exists because the std::this_thread::get_id() is much slower(espcially under VS 2013)
 */
inline size_t _thread_id()
{
#ifdef _WIN32
    return static_cast<size_t>(::GetCurrentThreadId());
#elif __linux__
#if defined(__ANDROID__) && defined(__ANDROID_API__) && (__ANDROID_API__ < 21)
#define SYS_gettid __NR_gettid
#endif
    return static_cast<size_t>(syscall(SYS_gettid));
#elif __FreeBSD__
    long tid;
    thr_self(&tid);
    return static_cast<size_t>(tid);
#else // Default to standard C++11 (OSX and other Unix)
    return static_cast<size_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
#endif
}

/*
 * Return current thread id as size_t (from thread local storage)
 */
inline const char *thread_id(const el::LogMessage *)
{
    // 64bit size_t can have at most 20 digits, change this if it's no longer true.
    static_assert(std::numeric_limits<size_t>::max() == 0xffffffffffffffff, "size_t is not 64bit");
    static const size_t SIZE_T_BUF_LEN = 20 + 1;

    struct tid_str_t
    {
        char c_str[SIZE_T_BUF_LEN];
        tid_str_t(size_t tid)
        {
            std::snprintf(c_str, SIZE_T_BUF_LEN, "%zu", tid);
        }
    };
    static thread_local const tid_str_t tid_str(_thread_id());
    return tid_str.c_str;
}

// logger ids
constexpr const auto kAllocTag = "alloc";
constexpr const auto kPerfTag = "performance";
constexpr const auto kDefTag = "default";

} // namespace logging

#define PerfLog(level) CLOG(level, logging::kPerfTag)
#define AllocLog(level) CLOG(level, logging::kAllocTag)
#define AllocVLog(level) CVLOG(level, logging::kAllocTag)

// Additional operator<< implementations
MAKE_LOGGABLE(std::exception_ptr, ep, os);

namespace executor {
class OpKernelDef;
class EvenlopDef;
}
MAKE_LOGGABLE(executor::OpKernelDef, c, os);
MAKE_LOGGABLE(executor::EvenlopDef, c, os);

namespace zmq {
class message_t;
class error_t;
}
MAKE_LOGGABLE(zmq::message_t, c, os);
MAKE_LOGGABLE(zmq::error_t, c, os);

namespace google {
namespace protobuf {
class Message;
}
}

MAKE_LOGGABLE(google::protobuf::Message, c, os);

/**
 * Generic operator<< for strong enum classes
 */
template<typename T>
using maybe_enum_t = typename std::enable_if_t<std::is_enum<T>::value, T>;

template<typename T>
MAKE_LOGGABLE(maybe_enum_t<T>, c, os)
{
    return os << static_cast<typename std::underlying_type_t<T>>(c);
}

/**
 * Generic operator<< for pointer types, excluding char*
 */
struct PtrPrintHelper { uint64_t value; };

template<typename T>
constexpr PtrPrintHelper as_hex(T *p)
{
    return { reinterpret_cast<uint64_t>(p) };
}

template<typename T>
constexpr PtrPrintHelper as_hex(const std::unique_ptr<T> &p)
{
    return { reinterpret_cast<uint64_t>(p.get()) };
}

template<typename T>
constexpr PtrPrintHelper as_hex(const std::shared_ptr<T> &p)
{
    return { reinterpret_cast<uint64_t>(p.get()) };
}

MAKE_LOGGABLE(PtrPrintHelper, helper, os);

#endif // LOGGING_H
