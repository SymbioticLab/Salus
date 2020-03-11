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

#ifndef SALUS_PLATFORM_LOGGING_H
#define SALUS_PLATFORM_LOGGING_H

#include "config.h"

#include "easylogging++.h"

#include <nlohmann/json.hpp>

#include <type_traits>
#include <optional>

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
        explicit tid_str_t(size_t tid)
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
constexpr const auto kOpTracing = "optracing";
constexpr const auto kDefTag = "default";
constexpr const auto kSMTag = "smtracing";

// logging configurations
struct Params
{
    std::optional<std::string> configFile;
    std::optional<int> verbosity;
    std::optional<std::string> vModules;
    std::optional<std::string> vLogFile;
    std::optional<std::string> pLogFile;
};
void initialize(const Params &params);

} // namespace logging

#define LogPerf() CLOG(TRACE, logging::kPerfTag)
#define LogAlloc() CLOG(TRACE, logging::kAllocTag)
#define LogOpTracing() CLOG(TRACE, logging::kOpTracing)
#define LogSMTracing() CLOG(TRACE, logging::kSMTag)

// Additional operator<< implementations
MAKE_LOGGABLE(std::exception_ptr, ep, os);

namespace executor {
class OpKernelDef;
class EvenlopDef;
}
MAKE_LOGGABLE(executor::OpKernelDef, c, os);
MAKE_LOGGABLE(executor::EvenlopDef, c, os);

namespace zmq {
class error_t;
}
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

inline auto as_json(nlohmann::json::initializer_list_t &&obj)
{
    return nlohmann::json(std::forward<nlohmann::json::initializer_list_t>(obj));
}

#endif // SALUS_PLATFORM_LOGGING_H
