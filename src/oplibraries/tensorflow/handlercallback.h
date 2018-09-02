/*
 * Copyright (c) 2018, peifeng <email>
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

#ifndef SALUS_OPLIB_TENSORFLOW_HANDLERCALLBACK_H
#define SALUS_OPLIB_TENSORFLOW_HANDLERCALLBACK_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "oplibraries/ioplibrary.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "utils/protoutils.h"
#include "utils/macros.h"

namespace salus::oplib::tensorflow {

struct HandlerCallback
{
    IOpLibrary::DoneCallback cb;
    ProtoPtr tfresp;
    void operator()(const Status &s) const;

    HandlerCallback() = default;

    HandlerCallback(IOpLibrary::DoneCallback cb, ProtoPtr tfresp)
        : cb(std::move(cb))
        , tfresp(std::move(tfresp))
    {
    }

    HandlerCallback(HandlerCallback &&other) noexcept
        : HandlerCallback(std::move(other.cb), std::move(other.tfresp))
    {
    }

    HandlerCallback &operator =(HandlerCallback &&other) noexcept
    {
        cb = std::move(other.cb);
        tfresp = std::move(other.tfresp);
        return *this;
    }

    SALUS_DISALLOW_COPY_AND_ASSIGN(HandlerCallback);
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_HANDLERCALLBACK_H
