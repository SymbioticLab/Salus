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

#include "memory.h"

#include <cstdlib>

void *mem::alignedAlloc(size_t size, int minimum_alignment)
{
    return aligned_alloc(minimum_alignment, size);
}

void mem::alignedFree(void *aligned_memory)
{
    // FUTURE: change to use mem::free after aligned_alloc is added to c++ standard.
    ::free(aligned_memory);
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
