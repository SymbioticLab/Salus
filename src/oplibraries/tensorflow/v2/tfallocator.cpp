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
#include "utils/macros.h"

namespace {
void checkMemory(void *ptr, size_t num_bytes)
{
    UNUSED(ptr);
    UNUSED(num_bytes);
    return;
#ifndef NDEBUG
    DEBUG("Checking memory at {:x} of size {}", reinterpret_cast<uint64_t>(ptr), num_bytes);
    uint8_t *pbegin = reinterpret_cast<uint8_t*>(ptr);
    uint8_t *pend = pbegin + num_bytes;
    for (auto p = pbegin; p != pend; ++p) {
        *p = 0xde;
        if (*p != 0xde) {
            ERR("Some wrong at address {:x}, which belongs to block {:x}",
                reinterpret_cast<uint64_t>(p), reinterpret_cast<uint64_t>(pbegin));
        }
    }
#endif
}

} // namespace

TFAllocator::TFAllocator(tensorflow::Allocator *other)
    : m_actualAlloc(other)
{
}

TFAllocator::~TFAllocator() = default;

std::string TFAllocator::Name()
{
    return "mock_tf";
}

bool TFAllocator::ShouldAllocateEmptyTensors()
{
    if (m_actualAlloc)
        return m_actualAlloc->ShouldAllocateEmptyTensors();
    return true;
}

void *TFAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
{
    void *ptr = nullptr;
    if (m_actualAlloc) {
        INFO("Using actual allocator: {}", m_actualAlloc->Name());
        ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes);
    } else {
        ptr = MemoryMgr::instance().allocate(alignment, num_bytes);
    }

    INFO("TFAllocator allocated {} bytes of memory at {:x} with alignment {}",
         num_bytes, reinterpret_cast<uint64_t>(ptr), alignment);

    checkMemory(ptr, num_bytes);
    return ptr;
}

void *TFAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
                               const tensorflow::AllocationAttributes& allocation_attr)
{
    void *ptr = nullptr;

    if (m_actualAlloc) {
        INFO("Using actual allocator: {}", m_actualAlloc->Name());
        ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes, allocation_attr);
        checkMemory(ptr, num_bytes);
    } else {
        ptr = AllocateRaw(alignment, num_bytes);
    }

    INFO("TFAllocator called for attributes {} of {} bytes of memory at {:x} with alignment {}",
         allocation_attr, num_bytes, reinterpret_cast<uint64_t>(ptr), alignment);
    return ptr;
}

void TFAllocator::DeallocateRaw(void *ptr)
{
    INFO("TFAllocator deallocating memory at {:x}", reinterpret_cast<uint64_t>(ptr));
    if (m_actualAlloc) {
        INFO("Using actual allocator: {}", m_actualAlloc->Name());
        m_actualAlloc->DeallocateRaw(ptr);
    } else {
        MemoryMgr::instance().deallocate(ptr);
    }
}
