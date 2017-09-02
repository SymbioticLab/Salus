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

#ifndef TFALLOCATOR_H
#define TFALLOCATOR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include <memory>

/**
 * @todo write docs
 */
class TFAllocator : public tensorflow::Allocator
{
public:
    TFAllocator(tensorflow::Allocator *other = nullptr);

    ~TFAllocator() override;

    std::string Name() override;
    void *AllocateRaw(size_t alignment, size_t num_bytes) override;
    void* AllocateRaw(size_t alignment, size_t num_bytes,
                      const tensorflow::AllocationAttributes& allocation_attr) override;

    void DeallocateRaw(void *ptr) override;
    bool ShouldAllocateEmptyTensors() override;

private:
    tensorflow::Allocator *m_actualAlloc;

    TF_DISALLOW_COPY_AND_ASSIGN(TFAllocator);
};

#endif // TFALLOCATOR_H
