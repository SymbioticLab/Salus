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

#include "platform/memory.h"

#include "config.h" // IWYU: keep

#include <cstdlib>

void *mem::alignedAlloc(int minimum_alignment, size_t size)
{
#if defined(HAS_CXX_ALIGNED_ALLOC)
    return std::aligned_alloc(minimum_alignment, size);
#else
    void *ptr = nullptr;
    // posix_memalign requires that the requested alignment be at least
    // sizeof(void*). In this case, fall back on malloc which should return
    // memory aligned to at least the size of a pointer.
    const int required_alignment = sizeof(void *);
    if (minimum_alignment < required_alignment)
        return malloc(size);
    int err = posix_memalign(&ptr, minimum_alignment, size);
    if (err != 0) {
        return nullptr;
    } else {
        return ptr;
    }
#endif // HAS_CXX_ALIGNED_ALLOC
}

void mem::alignedFree(void *aligned_memory)
{
#if defined(HAS_CXX_ALIGNED_ALLOC)
    free(aligned_memory);
#else
    ::free(aligned_memory);
#endif // HAS_CXX_ALIGNED_ALLOC
}

void *mem::malloc(size_t size)
{
    return std::malloc(size);
}

void *mem::realloc(void *ptr, size_t size)
{
    return std::realloc(ptr, size);
}

void mem::free(void *ptr)
{
    std::free(ptr);
}
