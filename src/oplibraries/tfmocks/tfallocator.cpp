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

#include "tfallocator.h"

#include "memorymgr/memorymgr.h"
#include "platform/logging.h"


TFAllocator::TFAllocator()
{

}

TFAllocator::~TFAllocator()
{

}

std::string TFAllocator::Name()
{
    return "mock_tf";
}

bool TFAllocator::ShouldAllocateEmptyTensors()
{
    return false;
}

void *TFAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
{
    if (num_bytes == 0) {
        INFO("Allocation for tracking purpose");
        num_bytes = 1;
    }

    auto ptr = MemoryMgr::instance().allocate(num_bytes, alignment);

    INFO("TFAllocator allocated {} bytes of memory at {:x} with alignment {}",
         num_bytes, reinterpret_cast<uint64_t>(ptr), alignment);

    return ptr;
}

void TFAllocator::DeallocateRaw(void *ptr)
{
    INFO("TFAllocator deallocating memory at {:x}", reinterpret_cast<uint64_t>(ptr));
    MemoryMgr::instance().deallocate(ptr);
}
