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
 */

#ifndef TENSORUTILS_H
#define TENSORUTILS_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include <memory>

/**
 * Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
 */
struct Entry
{
    Entry() = default;
    Entry(const Entry &other)
        : ref(other.ref)
        , ref_mu(other.ref_mu)
        , has_value(other.has_value)
        , val_field_is_set(other.val_field_is_set)
        , alloc_attr(other.alloc_attr)
        , alloc_ticket(other.alloc_ticket)
        , device_context(other.device_context)
        , device(other.device)
    {
        if (val_field_is_set) {
            val.Init(*other.val);
        }
    }
    ~Entry()
    {
        if (val_field_is_set)
            val.Destroy();
    }

    Entry &operator=(const Entry &other)
    {
        if (val_field_is_set) {
            val.Destroy();
        }
        ref = other.ref;
        ref_mu = other.ref_mu;
        has_value = other.has_value;
        val_field_is_set = other.val_field_is_set;
        alloc_attr = other.alloc_attr;
        alloc_ticket = other.alloc_ticket;
        device_context = other.device_context;
        device = other.device;
        if (val_field_is_set) {
            val.Init(*other.val);
        }
        return *this;
    }

    Entry &operator=(Entry &&other)
    {
        if (val_field_is_set) {
            val.Destroy();
        }
        ref = other.ref;
        ref_mu = other.ref_mu;
        has_value = other.has_value;
        val_field_is_set = other.val_field_is_set;
        alloc_attr = other.alloc_attr;
        alloc_ticket = other.alloc_ticket;
        device_context = other.device_context;
        device = std::move(other.device);
        if (val_field_is_set) {
            val.Init(std::move(*other.val));
        }
        return *this;
    }

    // Clears the <val> field.
    void ClearVal()
    {
        if (val_field_is_set) {
            val.Destroy();
            val_field_is_set = false;
            has_value = false;
        }
        // release device
        device.reset();
    }

    void Dereference()
    {
        {
            tf::mutex_lock l(*ref_mu);
            DCHECK(!val_field_is_set);
            val.Init(*ref);
            val_field_is_set = true;
        }
        ref = nullptr;
        ref_mu = nullptr;
    }

    tf::Tensor *MaybeDereference()
    {
        if (ref) {
            Dereference();
        }
        return val.get();
    }

    // A tensor value, if val_field_is_set.
    tf::ManualConstructor<tf::Tensor> val;

    tf::Tensor *ref = nullptr;   // A tensor reference.
    tf::mutex *ref_mu = nullptr; // mutex for *ref if ref is not nullptr.

    // Whether the value exists, either in <val> or <ref>.
    bool has_value = false;

    bool val_field_is_set = false;

    // The attributes of the allocator that creates the tensor.
    tf::AllocatorAttributes alloc_attr;
    // The ticket used to allocate the tensor
    uint64_t alloc_ticket;

    // Every entry carries an optional DeviceContext containing
    // Device-specific information about how the Tensor was produced.
    tf::DeviceContext *device_context = nullptr;
    std::shared_ptr<tf::Device> device = nullptr;

    // Whether this entry is expected to be used as ref
    bool expect_ref = false;
};

/**
 * Automatically dereference and move tensor to dstDevice if needed
 */
tf::Status derefMoveTensor(Entry &entry, const std::shared_ptr<tf::Device> &dstDevice,
                           tf::DeviceContext *dstCtx, const tf::AllocatorAttributes &attr,
                           const std::string &name = "");

#endif // TENSORUTILS_H
