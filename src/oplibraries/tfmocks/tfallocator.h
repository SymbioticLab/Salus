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

#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/platform/types.h>
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>

#include <memory>

/**
 * @todo write docs
 */
class TFAllocator : public tensorflow::Allocator
{
public:
    static std::shared_ptr<TFAllocator> New();

    ~TFAllocator() override;

    // bring up all AllocateRaw signiture from base class to surpass -Woverloaded-virtual
    using Allocator::AllocateRaw;

    std::string Name() override;
    void *AllocateRaw(size_t alignment, size_t num_bytes) override;
    void DeallocateRaw(void *ptr) override;
    bool ShouldAllocateEmptyTensors() override;

private:
    TFAllocator();

    TF_DISALLOW_COPY_AND_ASSIGN(TFAllocator);
};

#endif // TFALLOCATOR_H
