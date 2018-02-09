/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright 2018  Peifeng Yu <peifeng@umich.edu>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of
 * the License or (at your option) version 3 or any later version
 * accepted by the membership of KDE e.V. (or its successor approved
 * by the membership of KDE e.V.), which shall act as a proxy
 * defined in Section 14 of version 3 of the license.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SYMBIOTIC_SALUS_UTILS_TYPE_TRAITS_H
#define SYMBIOTIC_SALUS_UTILS_TYPE_TRAITS_H

#include <type_traits>
#include <tuple>

namespace symbiotic::salus {

template<typename... Args>
using arg_first_t = std::tuple_element_t<0, std::tuple<Args...>>;

template<typename... Args>
using arg_last_t = std::tuple_element_t<sizeof...(Args) - 1, std::tuple<Args...>>;

} // namespace symbiotic::salus::utils

#endif // SYMBIOTIC_SALUS_UTILS_TYPE_TRAITS_H
