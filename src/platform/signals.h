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

#ifndef SALUS_PLATFORM_SIGNALS_H
#define SALUS_PLATFORM_SIGNALS_H

#include <utility>

namespace signals {

enum class SignalAction
{
    Exit,
    Ignore,
};

void initialize();

std::pair<int, SignalAction> waitForTerminate();

using Handler = void (int);

void installSignalHandler(int sig, Handler);

const char *signalName(int sig);

}

#endif // SALUS_PLATFORM_SIGNALS_H
