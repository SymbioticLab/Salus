/*
 * Copyright (c) 2019, peifeng <email>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cudahook.h"
#include "cudahook_export.h"

#include "realdlsym.h"

#include <dlfcn.h>

#include <iostream>
#include <cstring>
#include <cstdlib>

using salus::real_dlsym;

/*
 * We need to give the pre-processor a chance to replace a function, such as:
 * cuMemAlloc => cuMemAlloc_v2
 */
#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

namespace salus {

CudaHook::CudaHook(const char *dl)
{
    // Load the libcuda.so library with RTLD_GLOBAL so we can hook the function calls
    m_handle = dlopen(dl, RTLD_LAZY | RTLD_GLOBAL);
    if (!m_handle) {
        std::cerr << "Error to open library " << dl << ": " << dlerror() << std::endl;
        std::exit(-1);
    }

#define USE_FUNC(funcname, ret, params, ...) \
    m_orig.funcname = func_cast<Fn_##funcname*>(real_dlsym(m_handle, CUDA_SYMBOL_STRING(funcname))); \
    if (!m_orig.funcname) { \
        std::cerr << "Error to find symbol " CUDA_SYMBOL_STRING(funcname) ": " << dlerror() << std::endl; \
        std::exit(-2); \
    }
#include "functions.def"

    auto envDebug = std::getenv("CUDA_HOOK_DEBUG");
    if (envDebug && envDebug[0] == '1') {
        m_debugging = true;
        std::cerr << "CUDA HOOK Library loaded." << std::endl;
    }
}

CudaHook &CudaHook::instance()
{
    static CudaHook hook("libcuda.so");
    return hook;
}

CudaHook::~CudaHook() {
    if (m_handle) {
        dlclose(m_handle);
    }
}

struct HookAccessor
{
    const salus::CudaHook &hook;

    bool debugging() const
    {
        return hook.m_debugging;
    }

    const auto &orig() const
    {
        return hook.m_orig;
    }

    const auto &pre() const
    {
        return hook.m_pre;
    }

    const auto &post() const
    {
        return hook.m_post;
    }
};

} // namespace salus


/*
 * Interposed Functions
 */
extern "C" {

/*
 * Other interposed functions
 */
#define USE_FUNC(funcname, ret, params, ...) \
    CUDAHOOK_EXPORT int funcname params \
    { \
        const salus::HookAccessor hook{salus::CudaHook::instance()}; \
        if (hook.pre().funcname) { \
            hook.pre().funcname(__VA_ARGS__); \
        } \
        if (hook.debugging()) { \
            std::cerr << "Hooked function " CUDA_SYMBOL_STRING(funcname) " is called\n";\
        } \
        auto res = hook.orig().funcname(__VA_ARGS__); \
        if (hook.post().funcname) { \
            hook.post().funcname(__VA_ARGS__); \
        } \
        return res; \
    }
#include "functions.def"

/*
 * We need to interpose dlsym since anyone using dlopen+dlsym to get the CUDA driver symbols will bypass
 * the hooking mechanism (this includes the CUDA runtime). Its tricky though, since if we replace the
 * real dlsym with ours, we can't dlsym() the real dlsym. To get around that, call the 'private'
 * libc interface called __libc_dlsym to get the real dlsym.
 */
CUDAHOOK_EXPORT void* dlsym(void *handle, const char *symbol) noexcept
{
    // Early out if not a CUDA driver symbol
    if (strncmp(symbol, "cu", 2) != 0) {
        return real_dlsym(handle, symbol);
    }

    const salus::HookAccessor hook{salus::CudaHook::instance()};
    if (hook.debugging()) {
        std::cerr << "Hooked dlsym: requesting " << symbol << "\n";
    }

#define USE_FUNC(funcname, ret, params, ...) \
    if (strcmp(symbol, CUDA_SYMBOL_STRING(funcname)) == 0) { \
        return reinterpret_cast<void*>(&funcname);\
    }
#include "functions.def"

    return real_dlsym(handle, symbol);
}

} // extern "C"
