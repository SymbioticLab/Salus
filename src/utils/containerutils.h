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

#ifndef SALUS_SSTL_CONTAINERUTILS_H
#define SALUS_SSTL_CONTAINERUTILS_H

#include <boost/range/adaptor/reversed.hpp>

#include <atomic>
#include <optional>

namespace sstl {

template<typename C>
auto optionalGet(const C &c, const typename C::key_type &k) -> std::optional<typename C::mapped_type>
{
    std::optional<typename C::mapped_type> res;
    auto it = c.find(k);
    if (it != c.end()) {
        res = it->second;
    }
    return res;
}

template<typename C>
auto optionalGet(const std::optional<C> &c, const typename C::key_type &k)
    -> std::optional<typename C::mapped_type>
{
    std::optional<typename C::mapped_type> res;
    auto it = c->find(k);
    if (it != c->end()) {
        res = it->second;
    }
    return res;
}

template<typename C>
auto getOrDefault(const C &c, const typename C::key_type &k, const typename C::mapped_type &defv) ->
    typename C::mapped_type
{
    auto it = c.find(k);
    if (it == c.end()) {
        return defv;
    }
    return it->second;
}

template<typename C>
auto getOrDefault(const std::optional<C> &c, const typename C::key_type &k,
                  const typename C::mapped_type &defv) -> typename C::mapped_type
{
    auto it = c->find(k);
    if (it == c->end()) {
        return defv;
    }
    return it->second;
}

template<typename Container, typename Predicate>
bool erase_if(Container &c, Predicate &&p)
{
    using std::begin;
    using std::end;

    auto itend = end(c);
    auto it = std::remove_if(begin(c), itend, std::forward<Predicate>(p));
    if (it == itend) {
        return false;
    }
    c.erase(it, itend);
    return true;
}

using boost::adaptors::reverse;

struct MutableAtom
{
    using value_type = std::atomic_uint_fast64_t;
    mutable value_type value{0};

    MutableAtom() = default;
    MutableAtom(MutableAtom &&other) noexcept
        : value(other.value.load())
    {
    }

    value_type &get() const
    {
        return value;
    }
};

} // namespace sstl

#endif // SALUS_SSTL_CONTAINERUTILS_H
