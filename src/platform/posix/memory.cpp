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
