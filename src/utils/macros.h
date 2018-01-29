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

#ifndef SALUS_UTILS_MACROS_H
#define SALUS_UTILS_MACROS_H

#include "config.h"

#define UNUSED(x) (void) (x)

#if !defined(HAS_CXX_ENUM_HASH)
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
#endif // HAS_CXX_ENUM_HASH

// GCC/Clang can be told that a certain branch is not likely to be taken (for
// instance, a CHECK failure), and use that information in static analysis.
// Giving it this information can help it optimize for the common case in
// the absence of better information (ie. -fprofile-arcs).
#if defined(HAS_CXX_BUILTIN_EXPECT)
#define SALUS_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define SALUS_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define SALUS_PREDICT_FALSE(x) (x)
#define SALUS_PREDICT_TRUE(x) (x)
#endif // HAS_CXX_BUILTIN_EXPECT

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define SALUS_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

constexpr std::size_t operator "" _sz (unsigned long long n) { return n; }

namespace utils {
template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}
} // namespace utils

#endif // SALUS_UTILS_MACROS_H
