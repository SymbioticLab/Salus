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

#ifndef SALUS_SSTL_TYPE_TRAITS_H
#define SALUS_SSTL_TYPE_TRAITS_H

#include <tuple>
#include <type_traits>

namespace sstl {

template<typename... Args>
using arg_first_t = std::tuple_element_t<0, std::tuple<Args...>>;

template<typename... Args>
using arg_last_t = std::tuple_element_t<sizeof...(Args) - 1, std::tuple<Args...>>;

} // namespace sstl

#endif // SALUS_SSTL_TYPE_TRAITS_H
