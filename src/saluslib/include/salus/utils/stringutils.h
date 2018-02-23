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

#ifndef SALUS_SSTL_STRINGUTILS_H
#define SALUS_SSTL_STRINGUTILS_H

#include <string>
#include <string_view>
#include <vector>

namespace sstl {

std::string bytesToHexString(const uint8_t *info, size_t infoLength, size_t maxLen = 20);

bool startsWith(std::string_view str, std::string_view prefix);

bool endsWith(std::string_view str, std::string_view postfix);

std::vector<std::string_view> splitsv(std::string_view self, std::string_view Separator);

std::vector<std::string_view> splitsv(std::string_view self, std::string_view::value_type Separator);

} // namespace sstl

#endif // SALUS_SSTL_STRINGUTILS_H
