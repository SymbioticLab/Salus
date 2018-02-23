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

#include "utils/stringutils.h"

#include <algorithm>
#include <string_view>
#include <vector>

namespace sstl {
std::string bytesToHexString(const uint8_t *info, size_t infoLength, size_t maxLen)
{
    static const char pszNibbleToHex[] = "0123456789ABCDEF";
    static const char ellipses[] = "...";
    static const size_t ellipsesLen = sizeof(ellipses) / sizeof(char);

    if (infoLength == 0 || !info) {
        return {};
    }

    std::string result(infoLength * 2, ' ');

    // TODO: we should be able to skip some iterations based on maxLen
    for (size_t i = 0; i < infoLength; i++) {
        auto nNibble = info[i] >> 4;
        result[2 * i] = pszNibbleToHex[nNibble];
        nNibble = info[i] & 0x0F;
        result[2 * i + 1] = pszNibbleToHex[nNibble];
    }

    if (maxLen > 0 && result.size() > maxLen) {
        if (maxLen <= ellipsesLen + 2) {
            result.erase(maxLen);
        } else {
            auto leading = (maxLen - ellipsesLen) / 2;
            auto ommitted = result.size() - (maxLen - ellipsesLen);
            result.replace(leading, ommitted, ellipses);
        }
    }

    return result;
}

bool startsWith(std::string_view str, std::string_view prefix)
{
    if (prefix.size() > str.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), str.begin());
}

bool endsWith(std::string_view str, std::string_view postfix)
{
    if (postfix.size() > str.size()) {
        return false;
    }
    return std::equal(postfix.begin(), postfix.end(), str.end() - postfix.size());
}

std::vector<std::string_view> splitsv(std::string_view self, std::string_view Separator)
{
    if (Separator.empty()) {
        return splitsv(self, 0);
    } else if (Separator.size() <= 1) {
        return splitsv(self, Separator[0]);
    }

    std::vector<std::string_view> Result;
    if (self.empty())
        return Result;

    std::string_view::size_type p, start = 0;
    while (true) {
        p = self.find(Separator, start);
        if (p == std::string_view::npos) {
            Result.emplace_back(self.substr(start, std::string_view::npos));
            return Result;
        } else {
            Result.emplace_back(self.substr(start, p - start));
            start = p + Separator.length();
        }
    }
}

std::vector<std::string_view> splitsv(std::string_view self, std::string_view::value_type Separator)
{
    std::vector<std::string_view> Result;
    if (self.empty())
        return Result;
    std::string_view::size_type p = 0, start = 0;

    if (Separator == 0) {
        for (p = 0; p < self.size(); ++p)
            Result.emplace_back(self.substr(p, 1));
        return Result;
    }
    while (true) {
        if (p >= self.size()) {
            Result.emplace_back(self.substr(start));
            return Result;
        }
        if (self[p] == Separator) {
            Result.emplace_back(self.substr(start, p - start));
            ++p;
            start = p;
        } else {
            ++p;
        }
    }
}
} // namespace sstl
