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

/*
 * Make sure tensorflow_headers is included first before
 * any other headers, so we can correctly override TF logging
 * with ours.
 */
#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "tfallocator.h"

#include "execution/executionengine.h"
#include "memorymgr/memorymgr.h"
#include "platform/logging.h"
#include "utils/macros.h"
#include "utils/threadutils.h"
#include "utils/stringutils.h"

namespace {
void checkMemory(void *ptr, size_t num_bytes)
{
#if !defined(NDEBUG)
    if (!VLOG_IS_ON(4)) {
        return;
    }
    VLOG(4) << "Checking memory at " << as_hex(ptr) << " of size " << num_bytes;
    uint8_t *pbegin = reinterpret_cast<uint8_t *>(ptr);
    uint8_t *pend = pbegin + num_bytes;
    for (auto p = pbegin; p != pend; ++p) {
        *p = 0xde;
        DCHECK_EQ(*p, 0xde) << "Something wrong at address " << as_hex(p)
                            << ", which belongs to block " << as_hex(pbegin);
    }
#else
    UNUSED(ptr);
    UNUSED(num_bytes);
    return;
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
    AllocLog(DEBUG) << "TFAllocator allocating " << num_bytes
                    << " bytes of memory with alignment " << alignment
                    << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc);

    void *ptr = nullptr;
    if (m_actualAlloc) {
        ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes);
    } else {
        ptr = MemoryMgr::instance().allocate(alignment, num_bytes);
    }

    AllocLog(INFO) << "TFAllocator allocated " << num_bytes
                   << " bytes of memory at " << as_hex(ptr) << " with alignment " << alignment
            << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc);

    checkMemory(ptr, num_bytes);
    return ptr;
}

void *TFAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
                               const tensorflow::AllocationAttributes &allocation_attr)
{
    auto attr(allocation_attr);

    // We should not retry on failure due to the restarting feature
    attr.no_retry_on_failure = true;

    AllocLog(DEBUG) << "TFAllocator allocating attributes " << attr << " of " << num_bytes
                    << " bytes of memory with alignment " << alignment
                    << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc);

    void *ptr = nullptr;
    if (m_actualAlloc) {
        ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes, attr);
        checkMemory(ptr, num_bytes);
    } else {
        ptr = MemoryMgr::instance().allocate(alignment, num_bytes);
    }

    AllocLog(INFO) << "TFAllocator called for attributes " << attr << " of " << num_bytes
                   << " bytes of memory at " << as_hex(ptr) << " with alignment " << alignment
                   << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc);
    return ptr;
}

void TFAllocator::DeallocateRaw(void *ptr)
{
    AllocLog(INFO) << "TFAllocator deallocating memory at " << as_hex(ptr)
                   << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc);

    if (m_actualAlloc) {
        m_actualAlloc->DeallocateRaw(ptr);
    } else {
        MemoryMgr::instance().deallocate(ptr);
    }
}

PerOpAllocator *PerOpAllocator::downcast(tf::Allocator *alloc)
{
    if (utils::startsWith(alloc->Name(), NamePrefix)) {
        return static_cast<PerOpAllocator*>(alloc);
    }
    return nullptr;
}

PerOpAllocator::PerOpAllocator(const std::shared_ptr<const ResourceContext> &rctx, tensorflow::Allocator *other)
    : m_rctx(rctx)
    , m_actualAlloc(other)
{
    assert(m_rctx);
    assert(m_actualAlloc);
    switch (m_rctx->spec().type) {
    case DeviceType::GPU:
        assert(m_actualAlloc->Name() == "GPU_0_bfc");
        break;
    default:
        assert(m_actualAlloc->Name() != "GPU_0_bfc");
        break;
    }
}

PerOpAllocator::PerOpAllocator(std::shared_ptr<const ResourceContext> &&rctx, tensorflow::Allocator *other)
    : m_rctx(std::move(rctx))
    , m_actualAlloc(other)
{
    assert(m_rctx);
    assert(m_actualAlloc);
}

PerOpAllocator::~PerOpAllocator() = default;

const std::string PerOpAllocator::NamePrefix = "PerOp_";

std::string PerOpAllocator::Name()
{
    return tf::strings::StrCat(NamePrefix, nameOrNull(m_actualAlloc));
}

void *PerOpAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
{
    AllocLog(DEBUG) << "TFAllocator allocating " << num_bytes
                    << " bytes of memory with alignment " << alignment
                    << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc);

    void *ptr = nullptr;
    if (auto scope = m_rctx->allocMemory(num_bytes)) {
        ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes);
        if (!ptr) {
            scope.rollback();
        }
        checkMemory(ptr, num_bytes);
    } else {
        // No enough memory
        AllocLog(DEBUG) << "TFAllocator failed to allocate.";
        return nullptr;
    }


    recordSize(ptr, num_bytes);

    if (!ptr) {
        return ptr;
    }

    Ref();

    AllocLog(INFO) << "TFAllocator allocated " << num_bytes
                   << " bytes of memory at " << as_hex(ptr) << " with alignment " << alignment
                   << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc)
                   << " with " << *m_rctx;

    return ptr;
}

void* PerOpAllocator::AllocateRaw(size_t alignment, size_t num_bytes, const tensorflow::AllocationAttributes& allocation_attr)
{
    auto attr(allocation_attr);
    // We should not retry on failure due to the restarting feature
    attr.no_retry_on_failure = true;

    AllocLog(DEBUG) << "TFAllocator allocating attributes " << attr << " of " << num_bytes
                    << " bytes of memory with alignment " << alignment
                    << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc);

    void *ptr = nullptr;
    if (auto scope = m_rctx->allocMemory(num_bytes)) {
        ptr = m_actualAlloc->AllocateRaw(alignment, num_bytes, attr);
        if (!ptr) {
            scope.rollback();
        }
        checkMemory(ptr, num_bytes);
    } else {
        // No enough memory
        AllocLog(DEBUG) << "TFAllocator failed to allocate.";
        return nullptr;
    }

    recordSize(ptr, num_bytes);

    if (!ptr) {
        return ptr;
    }

    Ref();

    AllocLog(INFO) << "TFAllocator called for attributes " << attr << " of " << num_bytes
                   << " bytes of memory at " << as_hex(ptr) << " with alignment " << alignment
                   << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc)
                   << " with " << *m_rctx;
    return ptr;
}

size_t PerOpAllocator::RequestedSize(void* ptr)
{
    return findSize(ptr);
}

tf::int64 PerOpAllocator::AllocationId(void* ptr)
{
    return reinterpret_cast<tf::int64>(ptr);
}

void PerOpAllocator::DeallocateRaw(void *ptr)
{
    auto num_bytes = RequestedSize(ptr);

    AllocLog(INFO) << "TFAllocator deallocating memory at " << as_hex(ptr)
                   << " size " << num_bytes
                   << " using allocator " << nameOrNull(m_actualAlloc) << "@" << as_hex(m_actualAlloc)
                   << " with " << *m_rctx;

    m_rctx->deallocMemory(num_bytes);
    m_actualAlloc->DeallocateRaw(ptr);

    Unref();
}

bool PerOpAllocator::ShouldAllocateEmptyTensors()
{
    return m_actualAlloc->ShouldAllocateEmptyTensors();
}

void PerOpAllocator::recordSize(void *ptr, size_t size)
{
    utils::Guard g(m_mu);
    m_lastFailedAllocSize = size;
    if (ptr)
        m_allocated[ptr] = size;
}

size_t PerOpAllocator::findSize(void *ptr)
{
    utils::Guard g(m_mu);
    auto it = m_allocated.find(ptr);
    if (it == m_allocated.end()) {
        return 0;
    }
    return it->second;
}
