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

#ifndef SALUS_SSTL_ENVUTILS_H
#define SALUS_SSTL_ENVUTILS_H

#include <boost/lexical_cast.hpp>

namespace sstl {

/**
 * @brief Read a value of type `T` from environment variable. Use `def` as default value in case of
 * error or missing value.
 *
 * @param env the name of the environment variable
 * @param def default value in case of error
 * @return the read value or default value
 */
template<typename T, typename R = T, typename = std::enable_if<std::is_same_v<char *, std::decay_t<T>>>>
R fromEnvVar(const char *env, const T &def)
{
    const char *env_var_val = std::getenv(env);
    if (!env_var_val) {
        return def;
    }
    if constexpr (std::is_same_v<R, std::string_view> || std::is_same_v<R, std::string>) {
        return env_var_val;
    } else {
        T res;
        if (!boost::conversion::try_lexical_convert(env_var_val, res)) {
            return def;
        }

        return res;
    }
}

inline const char *fromEnvVarStr(const char *env, const char *def)
{
    auto env_var_val = std::getenv(env);
    if (!env_var_val) {
        return def;
    }
    return env_var_val;
}

/**
 * @brief Read a value of type `T` from environment variable. Use `def` as default value in case of
 * error or missing value. The value is only read once from the environment variable. Later call
 * to this function simply returns the cached value.
 *
 * @tparam tag a cache tag that should be different for different callsite
 * @param env the name of the environment variable
 * @param def default value in case of error
 * @return the read value or default value
 */
template<typename tag, typename T>
T fromEnvVarCached(const char *env, const T &def)
{
    static T res = fromEnvVar(env, def);
    return res;
}

} // namespace sstl
#endif // SALUS_SSTL_ENVUTILS_H
