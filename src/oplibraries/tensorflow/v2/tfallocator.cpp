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
    uint8_t *pbegin = reinterpret_cast<uint8_t *>(ptr);
    uint8_t *pend = pbegin + num_bytes;
    for (auto p = pbegin; p != pend; ++p) {
        *p = 0xde;
        if (*p != 0xde) {
            ERR("Some wrong at address {:x}, which belongs to block {:x}", reinterpret_cast<uint64_t>(p),
                reinterpret_cast<uint64_t>(pbegin));
        }
    }
#endif
}

std::string nameOrNull(tf::Allocator *alloc)
{
    if (alloc)
        return alloc->Name();
    return "None";
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
    TRACE("TFAllocator allocating {} bytes of memory with alignment {} using allocator {}@{}", num_bytes,
         alignment, nameOrNull(m_actualAlloc), as_hex(m_actualAlloc));

    void *ptr = nullptr;
    if (m_actualAlloc) {
        ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes);
    } else {
        ptr = MemoryMgr::instance().allocate(alignment, num_bytes);
    }

    DEBUG("TFAllocator allocated {} bytes of memory at {} with alignment {} using allocator {}@{}", num_bytes,
         as_hex(ptr), alignment, nameOrNull(m_actualAlloc), as_hex(m_actualAlloc));

    checkMemory(ptr, num_bytes);
    return ptr;
}

void *TFAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
                               const tensorflow::AllocationAttributes &allocation_attr)
{
    auto attr(allocation_attr);

    // We should not retry on failure due to the restarting feature
    attr.no_retry_on_failure = true;

    TRACE("TFAllocator allocating attributes {} of {} bytes of memory with alignment {}"
         " using allocator {}@{}",
         attr, num_bytes, alignment, nameOrNull(m_actualAlloc), as_hex(m_actualAlloc));

    void *ptr = nullptr;
    if (m_actualAlloc) {
        ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes, attr);
        checkMemory(ptr, num_bytes);
    } else {
        ptr = MemoryMgr::instance().allocate(alignment, num_bytes);
    }

    DEBUG("TFAllocator called for attributes {} of {} bytes of memory at {} with alignment {}"
         " using allocator {}@{}",
         attr, num_bytes, as_hex(ptr), alignment, nameOrNull(m_actualAlloc),
         as_hex(m_actualAlloc));
    return ptr;
}

void TFAllocator::DeallocateRaw(void *ptr)
{
    DEBUG("TFAllocator deallocating memory at {} using allocator {}@{}", as_hex(ptr),
         nameOrNull(m_actualAlloc), as_hex(m_actualAlloc));
    if (m_actualAlloc) {
        m_actualAlloc->DeallocateRaw(ptr);
    } else {
        MemoryMgr::instance().deallocate(ptr);
    }
}

PerOpAllocator::PerOpAllocator(uint64_t ticket, const DeviceSpec &spec,
                               ResourceMonitor &resMon, tensorflow::Allocator *other)
    : m_ticket(ticket)
    , m_spec(spec)
    , m_resMon(resMon)
    , m_actualAlloc(other)
{
    assert(m_actualAlloc);
}

PerOpAllocator::~PerOpAllocator() = default;

std::string PerOpAllocator::Name()
{
    return tf::strings::StrCat("PerOp_", m_ticket, "_", nameOrNull(m_actualAlloc));
}

void *PerOpAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
{
    TRACE("TFAllocator allocating {} bytes of memory with alignment {} using allocator {}@{}", num_bytes,
         alignment, nameOrNull(m_actualAlloc), as_hex(m_actualAlloc));

    Resources res {
        {{ResourceType::MEMORY, m_spec}, num_bytes}
    };
    if (!m_resMon.allocate(m_ticket, res)) {
        // No enough memory
        TRACE("TFAllocator failed to allocate.");
        return nullptr;
    }

    auto ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes);
    checkMemory(ptr, num_bytes);

    if (ptr) {
        m_allocated[ptr] = num_bytes;
        Ref();
    }

    DEBUG("TFAllocator allocated {} bytes of memory at {} with alignment {} using allocator {}@{}", num_bytes,
         as_hex(ptr), alignment, nameOrNull(m_actualAlloc), as_hex(m_actualAlloc));

    return ptr;
}

void* PerOpAllocator::AllocateRaw(size_t alignment, size_t num_bytes, const tensorflow::AllocationAttributes& allocation_attr)
{
    auto attr(allocation_attr);
    // We should not retry on failure due to the restarting feature
    attr.no_retry_on_failure = true;

    TRACE("TFAllocator allocating attributes {} of {} bytes of memory with alignment {}"
         " using allocator {}@{}",
         attr, num_bytes, alignment, nameOrNull(m_actualAlloc), as_hex(m_actualAlloc));

    Resources res {
        {{ResourceType::MEMORY, m_spec}, num_bytes}
    };
    if (!m_resMon.allocate(m_ticket, res)) {
        // No enough memory
        TRACE("TFAllocator failed to allocate.");
        return nullptr;
    }

    auto ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes, attr);
    checkMemory(ptr, num_bytes);

    if (ptr) {
        m_allocated[ptr] = num_bytes;
        Ref();
    }

    DEBUG("TFAllocator called for attributes {} of {} bytes of memory at {} with alignment {}"
         " using allocator {}@{}",
         attr, num_bytes, as_hex(ptr), alignment, nameOrNull(m_actualAlloc),
         as_hex(m_actualAlloc));
    return ptr;
}

size_t PerOpAllocator::RequestedSize(void* ptr)
{
    auto it = m_allocated.find(ptr);
    if (it == m_allocated.end()) {
        return 0;
    }
    return it->second;
}

tf::int64 PerOpAllocator::AllocationId(void* ptr)
{
    return reinterpret_cast<tf::int64>(ptr);
}

void PerOpAllocator::DeallocateRaw(void *ptr)
{
    auto num_bytes = RequestedSize(ptr);

    DEBUG("TFAllocator deallocating memory at {} size {} using allocator {}@{}", as_hex(ptr),
          num_bytes, nameOrNull(m_actualAlloc), as_hex(m_actualAlloc));

    Resources res {
        {{ResourceType::MEMORY, m_spec}, num_bytes}
    };
    m_resMon.free(res);

    m_actualAlloc->DeallocateRaw(ptr);
    Unref();
}

bool PerOpAllocator::ShouldAllocateEmptyTensors()
{
    return m_actualAlloc->ShouldAllocateEmptyTensors();
}
