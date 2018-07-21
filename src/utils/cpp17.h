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

#ifndef SALUS_SSTL_CPP17_H
#define SALUS_SSTL_CPP17_H

#include <string>
#include <system_error>
#include <utility>

namespace sstl {

// TODO: use macro check

struct from_chars_result
{
    const char *ptr;
    std::error_code ec;
};

template<typename T>
from_chars_result from_chars(const char *first, const char *last, T &value, int base = 10) noexcept
{
    size_t pos;
    from_chars_result fcr{first, {}};
    try {
        auto val = std::stoll(std::string(first, last), &pos, base);
        value = static_cast<T>(val);
        fcr.ptr = first + pos;
    } catch (std::invalid_argument &ex) {
        fcr.ec = std::make_error_code(std::errc::invalid_argument);
    } catch (std::out_of_range &ex) {
        fcr.ec = std::make_error_code(std::errc::result_out_of_range);
    } catch (...) {
        // ignore
    }

    return fcr;
}

template<typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept
{
    return static_cast<typename std::underlying_type<E>::type>(e);
}

/**
 * @brief Check whether `val' is in `args'
 * @tparam T
 * @tparam Args
 * @param val
 * @param args
 * @return
 */
template<typename T, typename... Args>
constexpr bool is_in(const T &val, const Args &... args)
{
    if constexpr (sizeof...(args) == 0) {
        return false;
    } else {
        return [](const auto &... p) { return (... || p); }((val == args)...);
    }
}

namespace detial {
template<class Tup, class Func, std::size_t... Is>
constexpr void static_for_impl(Tup &&t, Func &&f, std::index_sequence<Is...>)
{
    (f(std::integral_constant<std::size_t, Is>{}, std::get<Is>(std::forward<Tup>(t))), ...);
}
} // namespace detial

/**
 * @brief static for loop over a tuple
 * @tparam T
 * @tparam Func
 * @param t the tuple to loop over
 * @param f the functor, must be generic if the tuple is heterogeneous.
 * It has the signature functor(size_t index, T element)
 */
template<class... T, class Func>
constexpr void static_for(std::tuple<T...> &&t, Func &&f)
{
    detial::static_for_impl(std::forward<decltype(t)>(t), std::forward<Func>(f),
                            std::make_index_sequence<sizeof...(T)>{});
}

} // namespace sstl

#endif // SALUS_SSTL_CPP17_H
