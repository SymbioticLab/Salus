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

#ifndef MACROS_H
#define MACROS_H

#include "config.h"

#define UNUSED(x) (void) (x)

#ifndef HAS_CXX_ENUM_HASH
namespace std {
template<class E>
class hash
{
    using sfinae = typename std::enable_if<std::is_enum<E>::value, E>::type;

public:
    size_t operator()(const E &e) const
    {
        return std::hash<typename std::underlying_type<E>::type>()(e);
    }
};
} // namespace std
#endif // CXX_HAS_ENUM_HASH

#endif // MACROS_H
