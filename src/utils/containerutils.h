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

#ifndef CONTAINERUTILS_H
#define CONTAINERUTILS_H

#include "utils/cpp17.h"

#include <boost/range/adaptor/reversed.hpp>

namespace utils {

template<typename C, typename K, typename V>
V getOrDefault(C &c, const K &k, const V &defv)
{
    auto it = c.find(k);
    if (it == c.end()) {
        return defv;
    }
    return it->second;
}

template<typename C, typename K, typename V>
V getOrDefault(optional<C> &c, const K &k, const V &defv)
{
    auto it = c->find(k);
    if (it == c->end()) {
        return defv;
    }
    return it->second;
}

using boost::adaptors::reverse;

} // namespace utils

#endif // CONTAINERUTILS_H
