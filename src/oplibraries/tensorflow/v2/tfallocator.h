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

#include "execution/resources.h"

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include <memory>

class ResourceContext;

class TFAllocator : public tensorflow::Allocator
{
public:
    explicit TFAllocator(tensorflow::Allocator *other = nullptr);

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

class PerOpAllocator : public tensorflow::Allocator, public tensorflow::core::RefCounted
{
public:
    static const std::string NamePrefix;

    static PerOpAllocator *downcast(tf::Allocator *);

    explicit PerOpAllocator(const std::shared_ptr<ResourceContext> &rctx, tensorflow::Allocator *other);
    explicit PerOpAllocator(std::shared_ptr<ResourceContext> &&rctx, tensorflow::Allocator *other);

    ~PerOpAllocator() override;

    std::string Name() override;
    void *AllocateRaw(size_t alignment, size_t num_bytes) override;
    void* AllocateRaw(size_t alignment, size_t num_bytes,
                      const tensorflow::AllocationAttributes& allocation_attr) override;

    void DeallocateRaw(void *ptr) override;
    bool ShouldAllocateEmptyTensors() override;

    bool TracksAllocationSizes() override { return true; }

    size_t RequestedSize(void* ptr) override;

    tf::int64 AllocationId(void* ptr) override;

    const ResourceContext &resourceContext() const { return *m_rctx; }

private:
    void recordSize(void *ptr, size_t size);
    size_t findSize(void *ptr);

    std::shared_ptr<ResourceContext> m_rctx;

    tensorflow::Allocator *m_actualAlloc;

    std::mutex m_mu;
    std::unordered_map<void*, size_t> m_allocated;

    TF_DISALLOW_COPY_AND_ASSIGN(PerOpAllocator);
};

#endif // TFALLOCATOR_H
