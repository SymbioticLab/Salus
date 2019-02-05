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
