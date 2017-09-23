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

#ifndef UTILS_CPP17_H
#define UTILS_CPP17_H

#include <boost/optional.hpp>
#include <boost/any.hpp>

#include <system_error>
#include <string>

namespace utils {

using boost::optional;

using boost::any;
using boost::any_cast;

// TODO: use macro check

struct from_chars_result {
    const char* ptr;
    std::error_code ec;
};

template<typename T>
from_chars_result from_chars(const char* first, const char* last,
                             T& value, int base = 10) noexcept
{
    size_t pos;
    from_chars_result fcr {first, {}};
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

} // namespace utils

#endif // UTILS_CPP17_H
