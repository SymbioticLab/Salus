/*
 * Copyright (c) 2018, peifeng <email>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "oplibraries/tensorflow/device/sessionallocator.h"

namespace salus::oplib::tensorflow {

SessionAllocator::SessionAllocator(const std::string &sess, sstl::not_null<tf::Allocator *> base)
    : ForwardingAllocator(base)
    , m_sessHandle(sess)
{
}

SessionAllocator::~SessionAllocator() = default;

namespace {
std::string maybeMemmap(tf::Allocator *alloc)
{
    if (alloc->Name().find("dbfc") != std::string::npos) {
        if (auto dbfc = dynamic_cast<tf::GPUDoubleBFCAllocator*>(alloc)) {
            return dbfc->GenerateMemoryMap();
        }
    }
    return {};
}
} // namespace

void SessionAllocator::postAllocation(void *ptr, size_t alignment, size_t num_bytes,
                                      const tf::AllocationAttributes &)
{
    if (!ptr) {
        return;
    }
    UNUSED(maybeMemmap);
    LogAlloc() << "event: alloc "
               << nlohmann::json({
                      {"ptr", reinterpret_cast<uint64_t>(ptr)},
                      {"sess", m_sessHandle},
                      {"size", num_bytes},
                      {"alignment", alignment},
                      {"allocator", Name()},
//                      {"memmap", maybeMemmap(base())},
                  });
}

void SessionAllocator::preDeallocation(void *ptr)
{
    LogAlloc() << "event: dealloc "
               << nlohmann::json({
                      {"ptr", reinterpret_cast<uint64_t>(ptr)},
                      {"sess", m_sessHandle},
                      {"size", RequestedSize(ptr)},
                      {"allocator", Name()},
//                      {"memmap", maybeMemmap(base())},
                  });
}

} // namespace salus::oplib::tensorflow
