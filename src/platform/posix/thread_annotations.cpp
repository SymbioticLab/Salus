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
#include "platform/thread_annotations.h"

#include <pthread.h>

#include <string>

namespace salus::threading {

void set_thread_name(std::string_view name)
{
#if defined(__GLIBC__)
    // maximum length is 16 (including \0), longer than that casues error
    constexpr auto MAX_THREAD_NAME_LENGTH = 16 - 1;
    pthread_setname_np(pthread_self(), std::string(name.substr(0, MAX_THREAD_NAME_LENGTH)).c_str());
#elif defined(__APPLE__)
    pthread_setname_np(std::string(name).c_str());
#else
#error unsupported platform for POSIX!
#endif
}

} // namespace salus::threading
