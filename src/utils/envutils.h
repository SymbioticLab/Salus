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

#ifndef ENVUTILS_H
#define ENVUTILS_H

namespace utils {

/**
 * @brief Read a value of type `T` from environment variable. Use `def` as default value in case of
 * error or missing value.
 *
 * @param env the name of the environment variable
 * @param def default value in case of error
 * @return the read value or default value
 */
template<typename T>
T fromEnvVar(const char *env, const T &def);

/**
 * @brief Read a value of type `T` from environment variable. Use `def` as default value in case of
 * error or missing value. The value is only read once from the environment variable. Later call
 * to this function simply returns the cached value.
 *
 * @param env the name of the environment variable
 * @param def default value in case of error
 * @return the read value or default value
 */
template<typename T>
T fromEnvVarCached(const char *env, const T &def);

} // namespace utils
#endif // ENVUTILS_H
