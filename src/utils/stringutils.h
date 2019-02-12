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

#ifndef SALUS_SSTL_STRINGUTILS_H
#define SALUS_SSTL_STRINGUTILS_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace sstl {

std::string bytesToHexString(const uint8_t *info, size_t infoLength, size_t maxLen = 20);

bool startsWith(const std::string &str, const std::string &prefix);

bool endsWith(const std::string &str, const std::string &postfix);

} // namespace sstl

#endif // SALUS_SSTL_STRINGUTILS_H
