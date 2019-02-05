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

#ifndef MEMORY_H
#define MEMORY_H

#include <cstddef>

namespace mem {

// Aligned allocation/deallocation. `minimum_alignment` must be a power of 2
// and a multiple of sizeof(void*).
void *alignedAlloc(int minimum_alignment, size_t size);
void alignedFree(void *aligned_memory);

void *malloc(size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);

} // namespace mem

#endif // MEMORY_H
