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
