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

std::string utils::bytesToHexString(const uint8_t *info, size_t infoLength)
{
    static const char *pszNibbleToHex = {"0123456789ABCDEF"};

    if (infoLength <= 0 || !info) {
        return {};
    }

    std::string result(infoLength * 2, ' ');

    for (int i = 0; i < infoLength; i++) {
        int nNibble = info[i] >> 4;
        result[2 * i] = pszNibbleToHex[nNibble];
        nNibble = info[i] & 0x0F;
        result[2 * i + 1] = pszNibbleToHex[nNibble];
    }

    return result;
}
