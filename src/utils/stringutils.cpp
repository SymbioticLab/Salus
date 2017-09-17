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

#include "stringutils.h"

#include <algorithm>

namespace utils {
std::string bytesToHexString(const uint8_t *info, size_t infoLength, size_t maxLen)
{
    static const char pszNibbleToHex[] = "0123456789ABCDEF";
    static const char ellipses[] = "...";
    static const size_t ellipsesLen = sizeof(ellipses) / sizeof(char);

    if (infoLength <= 0 || !info) {
        return {};
    }

    std::string result(infoLength * 2, ' ');

    // TODO: we should be able to skip some iterations based on maxLen
    for (size_t i = 0; i < infoLength; i++) {
        int nNibble = info[i] >> 4;
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

bool startsWith(const std::string &str, const std::string &prefix)
{
    if (prefix.size() > str.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), str.begin());
}

} // namespace utils
