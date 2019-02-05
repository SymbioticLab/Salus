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

#ifndef PROTOS_H
#define PROTOS_H

// Force to use non-debug version of protobuf map, which changes its hashing function
// according to debug state, causing problems when two libraries both use protobuf, but
// only one of them is built with debug. Then passing a map from one library to the other
// becomes impossible because values inserted using one hashing function can't be found
// using another hashing function.
#ifndef NDEBUG
#define NDEBUG
#define NEED_UNDEF_NDEBUG
#endif

#include "executor.pb.h"
#include "tfoplibrary.pb.h"

#ifdef NEED_UNDEF_NDEBUG
#undef NDEBUG
#undef NEED_UNDEF_NDEBUG
#endif

#endif // PROTOS_H
