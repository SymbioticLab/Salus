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

#include "config.h"

#include <Windows.h>

#include <cstdlib>

void *mem::alignedAlloc(int minimum_alignment, size_t size)
{
#if HAS_CXX_ALIGNED_ALLOC
    return std::aligned_alloc(minimum_alignment, size);
#else
    return _aligned_malloc(size, minimum_alignment);
#endif
}

void mem::alignedFree(void *aligned_memory)
{
#if HAS_CXX_ALIGNED_ALLOC
    free(aligned_memory);
#else
    _aligned_free(aligned_memory);
#endif
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
