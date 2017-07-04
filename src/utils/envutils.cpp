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

#include "envutils.h"

#include <boost/lexical_cast.hpp>

template<typename T>
T utils::fromEnvVar(const char *env, const T &def)
{
    const char* env_var_val = std::getenv(env);
    if (!env_var_val) {
        return def;
    }
    T res;
    if (!boost::conversion::try_lexical_convert(env_var_val, res))
        return def;

    return res;
}

template<typename T>
T utils::fromEnvVarCached(const char *env, const T &def)
{
    static T res = fromEnvVar(env, def);
    return res;
}
