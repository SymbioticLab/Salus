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

#include <boost/range/adaptor/reversed.hpp>

#include <atomic>
#include <optional>

namespace symbiotic::salus {

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
    MutableAtom(MutableAtom &&other)
        : value(other.value.load())
    {
    }

    value_type &get() const
    {
        return value;
    }
};

} // namespace symbiotic::salus

#endif // CONTAINERUTILS_H
