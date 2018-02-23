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
 */

#ifndef SALUS_OPLIB_TENSORFLOW_PROTOS_H
#define SALUS_OPLIB_TENSORFLOW_PROTOS_H

// Force to use non-debug version of protobuf map, which changes its hashing function
// according to debug state, causing problems when two libraries both use protobuf, but
// only one of them is built with debug. Then passing a map from one library to the other
// becomes impossible because values inserted using one hashing function can't be found
// using another hashing function.
#ifndef NDEBUG
#define NDEBUG
#define NEED_UNDEF_NDEBUG
#endif

#include "tfoplibrary.pb.h"

#ifdef NEED_UNDEF_NDEBUG
#undef NDEBUG
#undef NEED_UNDEF_NDEBUG
#endif

#endif // SALUS_OPLIB_TENSORFLOW_PROTOS_H
