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

#ifndef SALUS_OPLIB_TENSORFLOW_TENSORUTILS_H
#define SALUS_OPLIB_TENSORFLOW_TENSORUTILS_H

/*
 * Make sure tensorflow_headers is included first before
 * any other headers, so we can correctly override TF logging
 * with ours.
 */
#include "oplibraries/tensorflow/tensorflow_headers.h"

#include <boost/intrusive/list_hook.hpp>
#include <boost/thread/shared_mutex.hpp>

#include <memory>
#include <unordered_map>

namespace salus::oplib::tensorflow {
struct Entry;
struct TensorBufferTree
    : public boost::intrusive::list_base_hook<boost::intrusive::link_mode<boost::intrusive::auto_unlink>>
{
    tf::TensorBuffer *root_buf = nullptr;
    uint64_t ticket;
    bool paged_out = false;
    boost::upgrade_mutex buf_mu;

    std::vector<Entry *> roots;
    std::unordered_map<tf::TensorBuffer *, std::vector<Entry *>> subs;

    bool empty() const
    {
        size_t size = 0;
        for (auto &sub : subs) {
            size += sub.second.size();
        }
        return roots.empty() && size == 0;
    }
};

/**
 * Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
 * TODO: use variant to hold either a reference or value
 */
struct Entry
{
    Entry() = default;
    Entry(const Entry &other)
        : ref(other.ref)
        , ref_mu(other.ref_mu)
        , has_value(other.has_value)
        , val_field_is_set(other.val_field_is_set)
    {
        if (val_field_is_set) {
            val.Init(*other.val);
        }
        CopyProperties(other);
    }

    ~Entry()
    {
        if (val_field_is_set)
            val.Destroy();
        // reset all pointers to be safe
        has_value = false;
        ref = nullptr;
        ref_mu = nullptr;
        val_field_is_set = false;
        alloc_ticket = 0xdeadbeef;
        alloc_tree = nullptr;
        device_context = nullptr;
        device = nullptr;
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
        if (val_field_is_set) {
            val.Init(*other.val);
        }
        CopyProperties(other);
        return *this;
    }

    Entry &operator=(Entry &&other) noexcept
    {
        if (val_field_is_set) {
            val.Destroy();
        }
        ref = other.ref;
        ref_mu = other.ref_mu;
        has_value = other.has_value;
        val_field_is_set = other.val_field_is_set;
        if (val_field_is_set) {
            val.Init(std::move(*other.val));
        }
        CopyProperties(other);
        return *this;
    }

    void CopyProperties(const Entry &other)
    {
        alloc_attr = other.alloc_attr;
        alloc_ticket = other.alloc_ticket;
        alloc_tree = other.alloc_tree;
        device_context = other.device_context;
        device = other.device;
    }

    void CopyProperties(Entry &&other)
    {
        alloc_attr = other.alloc_attr;
        alloc_ticket = other.alloc_ticket;
        alloc_tree = other.alloc_tree;
        device_context = other.device_context;
        device = std::move(other.device);
    }

    // Clears the <val> field.
    void ClearVal()
    {
        if (val_field_is_set) {
            val.Destroy();
            val_field_is_set = false;
        }
        // release device
        device.reset();
        has_value = false;
    }

    void Dereference()
    {
        if (alloc_tree) {
            VLOG(3) << "Dereferencing entry " << as_hex(this) << " ref " << as_hex(ref) << " buffer "
                    << as_hex(tf::remote::PagingHelper::bufferOf(*ref)) << " of ticket " << alloc_tree->ticket;
        } else {
            VLOG(3) << "Dereferencing entry " << as_hex(this) << " ref " << as_hex(ref) << " buffer "
                    << as_hex(tf::remote::PagingHelper::bufferOf(*ref)) << " of ticket null";
        }

        {
            tf::mutex_lock l(*ref_mu);
            DCHECK(!val_field_is_set);
            val.Init(*ref);
            val_field_is_set = true;
        }
        ref = nullptr;
        ref_mu = nullptr;
    }

    void MaybeDereference()
    {
        if (ref) {
            Dereference();
        }
    }

    tf::Tensor *RefOrVal()
    {
        if (ref) {
            return ref;
        }
        return val.get();
    }

    template<typename... Args>
    void SetVal(Args... args)
    {
        if (val_field_is_set) {
            val.Destroy();
        }
        ref = nullptr;
        ref_mu = nullptr;
        val.Init(std::forward<Args>(args)...);
        val_field_is_set = true;
        has_value = true;
    }

    struct MaybeLock
    {
        explicit MaybeLock(Entry *en)
            : mu(en->ref_mu)
        {
            if (mu) {
                mu->lock();
            }
        }
        ~MaybeLock()
        {
            if (mu) {
                mu->unlock();
            }
        }

    private:
        tf::mutex *mu;
    };

    // A tensor value, if val_field_is_set.
    tf::ManualConstructor<tf::Tensor> val;

    tf::Tensor *ref = nullptr;   // A tensor reference.
    tf::mutex *ref_mu = nullptr; // mutex for *ref if ref is not nullptr.

    // Whether the value exists, either in <val> or <ref>.
    bool has_value = false;

    bool val_field_is_set = false;

    // The attributes of the allocator that creates the tensor.
    tf::AllocatorAttributes alloc_attr;
    // The allocation ticket
    uint64_t alloc_ticket;
    // The buffer tree used to allocate the tensor
    TensorBufferTree *alloc_tree = nullptr;

    // Every entry carries an optional DeviceContext containing
    // Device-specific information about how the Tensor was produced.
    tf::DeviceContext *device_context = nullptr;
    std::shared_ptr<tf::Device> device = nullptr;
};

class PerTaskDevice;
/**
 * Automatically dereference and move tensor to dstDevice if needed
 */
tf::Status derefMoveTensor(Entry &entry, const std::shared_ptr<PerTaskDevice> &dstDevice,
                           tf::DeviceContext *dstCtx, const tf::AllocatorAttributes &attr,
                           const std::string &name = "");

/**
 * Move tensor to dstDevice.
 * Prerequest: entry.ref_mu locked if not null
 */
tf::Status moveTensor(Entry &entry, const std::shared_ptr<PerTaskDevice> &dstDevice,
                      tf::DeviceContext *dstCtx, const tf::AllocatorAttributes &attr,
                      const std::string &name = "");

tf::Status moveTensorTree(TensorBufferTree &, const std::shared_ptr<PerTaskDevice> &dstDevice);

} // namespace salus::oplib::tensorflow
#endif // SALUS_OPLIB_TENSORFLOW_TENSORUTILS_H
