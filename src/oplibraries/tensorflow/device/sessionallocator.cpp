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
