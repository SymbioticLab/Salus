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

#ifndef SALUS_OPLIB_TENSORFLOW_SESSIONALLOCATOR_H
#define SALUS_OPLIB_TENSORFLOW_SESSIONALLOCATOR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "oplibraries/tensorflow/device/shadowdevices.h"

namespace salus::oplib::tensorflow {

class SessionAllocator : public ForwardingAllocator
{
public:

    explicit SessionAllocator(const std::string &sess, sstl::not_null<tf::Allocator*> base);

    ~SessionAllocator() override;

protected:
    void postAllocation(void *ptr, size_t alignment, size_t num_bytes,
                        const tf::AllocationAttributes &allocation_attr) override;
    void preDeallocation(void *ptr) override;

private:
    std::string m_sessHandle;
};

} // namespace salus::oplib::tensorflow
#endif // SALUS_OPLIB_TENSORFLOW_SESSIONALLOCATOR_H
