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

#include "stringutils.h"

#include <algorithm>

namespace sstl {
std::string bytesToHexString(const uint8_t *info, size_t infoLength, size_t maxLen)
{
    constexpr char pszNibbleToHex[] = "0123456789ABCDEF";
    constexpr char ellipses[] = "...";
    constexpr size_t ellipsesLen = sizeof(ellipses) / sizeof(char);

    if (infoLength == 0 || !info) {
        return {};
    }

    std::string result(infoLength * 2, ' ');

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

bool endsWith(const std::string &str, const std::string &postfix)
{
    if (postfix.size() > str.size()) {
        return false;
    }
    return std::equal(postfix.begin(), postfix.end(), str.end() - postfix.size());
}

} // namespace sstl
