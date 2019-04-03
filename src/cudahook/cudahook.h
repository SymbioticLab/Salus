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

#ifndef SALUS_CUDAHOOK_H
#define SALUS_CUDAHOOK_H

#include <functional>

namespace salus {

struct HookAccessor;

/*
 * Typedefs of function types
 */
#define USE_FUNC(funcname, ret, params, ...) using Fn_##funcname = ret params;
#include "functions.def"

/**
 * @brief Callback structure for each of hooked function
 */
struct HookedFunctions
{
#define USE_FUNC(funcname, ret, params, ...) \
        std::function<Fn_##funcname> funcname = nullptr;
#include "functions.def"
};

class CudaHook
{
    HookedFunctions m_orig;

    HookedFunctions m_pre;
    HookedFunctions m_post;

    void *m_handle = nullptr;

    bool m_debugging = false;

    explicit CudaHook(const char *dl);

    friend struct HookAccessor;

public:
    ~CudaHook();

    static CudaHook &instance();

    HookedFunctions &post()
    {
        return m_post;
    }

    HookedFunctions &pre()
    {
        return m_pre;
    }
};

} // salus

#endif // SALUS_CUDAHOOK_H
