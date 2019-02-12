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

#ifndef SALUS_EXEC_ALLOCATIONLISTENER_H
#define SALUS_EXEC_ALLOCATIONLISTENER_H

#include "resources/resources.h"

namespace salus {
class AllocationListener
{
public:
    virtual ~AllocationListener() = default;
    /**
     * @brief Should be called by ResourceMonitor
     * @param ticket
     * @param res
     */
    virtual void notifyAlloc(uint64_t graphId, uint64_t ticket, const ResourceTag &rt, size_t num) = 0;

    /**
     * @brief Called when dealloc happens
     * @param ticket the ticket used
     * @param rt resource tag
     * @param num amount
     * @param last if this was the last allocation from ticket
     */
    virtual void notifyDealloc(uint64_t graphId, uint64_t ticket, const ResourceTag &rt, size_t num, bool last) = 0;
};

} // namespace salus

#endif // SALUS_EXEC_ALLOCATIONLISTENER_H
