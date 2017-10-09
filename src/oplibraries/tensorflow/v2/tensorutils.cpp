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

/*
 * Make sure tensorflow_headers is included first before
 * any other headers, so we can correctly override TF logging
 * with ours.
 */
#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "tensorutils.h"

#include "oplibraries/tensorflow/v2/peropallocdevice.h"
#include "execution/executionengine.h"

tensorflow::Status moveTensor(Entry &entry, const std::shared_ptr<PerOpAllocDevice> &dstDevice,
                              tensorflow::DeviceContext *dstCtx,
                              const tensorflow::AllocatorAttributes &attr, const std::string &name)
{
    auto input = entry.RefOrVal();

    tf::Tensor copy(dstDevice->GetAllocator(attr), input->dtype(), input->shape());

    if (!dstCtx) {
        // Copied from OpKernelContext::op_device_context
        auto dev_info = dstDevice->tensorflow_gpu_device_info();
        if (dev_info)
            dstCtx = dev_info->default_context;
    }

    VLOG(2) << "Src dev context " << as_hex(entry.device_context)
            << ", dst dev context " << as_hex(dstCtx)
            << ", source tensor buffer addr: " << as_hex(input->tensor_data().data())
            << ", target tensor buffer addr: " << as_hex(copy.tensor_data().data());

    tf::Status ok;
    tf::Notification n;
    tf::CopyTensor::ViaDMA(name, entry.device_context, dstCtx, entry.device.get(), dstDevice.get(),
                           entry.alloc_attr, attr, input, &copy, [&n, &ok](auto status) {
                               ok = status;
                               n.Notify();
                           });
    n.WaitForNotification();

    if (!ok.ok()) {
        return ok;
    }

    // Note copy is stack allocated, we need to move it back into entry
    if (entry.ref) {
        *input = std::move(copy);
    } else {
        entry.SetVal(std::move(copy));
    }

    entry.alloc_attr = attr;
    entry.device_context = dstCtx;
    entry.device = dstDevice;

    return tf::Status::OK();
}

bool moveTensorTree(TensorBufferTree &tree, const std::shared_ptr<PerOpAllocDevice> &dstDevice)
{
    if (tree.roots.empty()) {
        return false;
    }

    const auto oldRoot = tree.root_buf;
    const auto oldTicket = tree.ticket;

    tree.ticket = dstDevice->resourceContext().ticket();

    if (oldRoot == nullptr) {
        return true;
    }

    auto oldCount = tf::remote::PagingHelper::refCountOf(*oldRoot);
    VLOG(2) << "Moving tensor buffer " << as_hex(oldRoot) << " (count " << oldCount
            << ") with ticket " << oldTicket;

    oldRoot->Ref();

    std::unordered_set<tf::Tensor*> movedReferences;
    Entry *firstEntry = nullptr;
    tf::TensorBuffer *newRoot = nullptr;
    // Firstly page out root buffer
    for (auto entry : tree.roots) {
        if (!newRoot) {
            // only need to actually move the first in roots
            VLOG(3) << "    Actually move first in roots: entry " << as_hex(entry)
                    << " (ref " << as_hex(entry->ref) << ") with ticket " << oldTicket;
            auto ok = moveTensor(*entry, dstDevice, nullptr, {},
                                 tf::strings::StrCat("Paging tensor of ticket ", oldTicket));
            if (!ok.ok()) {
                LOG(ERROR) << "Error when paging: " << ok;
                return false;
            }
            newRoot = tf::remote::PagingHelper::bufferOf(*(entry->val.get()));
            firstEntry = entry;

            if (entry->ref) {
                movedReferences.insert(entry->ref);
            }
            continue;
        }
        VLOG(3) << "    Move other tensors of same root: entry " << as_hex(entry)
                << " (ref " << as_hex(entry->ref) << ") with ticket " << oldTicket;

        entry->CopyProperties(*firstEntry);
        // only one reference entry need to be moved
        if (entry->ref && movedReferences.count(entry->ref) > 0) {
            continue;
        }
        VLOG(3) << "    Move other tensors of same root: ref " << as_hex(entry->ref)
                << " ticket " << oldTicket << " not yet moved or this is value";

        auto t = tf::remote::PagingHelper::cloneWithNewBuffer(*entry->RefOrVal(), newRoot);
        if (entry->ref) {
            *entry->ref = std::move(t);
            movedReferences.insert(entry->ref);
        } else {
            entry->SetVal(std::move(t));
        }
    }

    DCHECK(newRoot);
    tree.root_buf = newRoot;

    // Secondly re-target sub buffers to new root
    // and update subs
    std::unordered_map<tf::TensorBuffer*, std::vector<Entry*>> newSubs;
    newSubs.reserve(tree.subs.size());
    for (auto &pp : tree.subs) {
        auto oldSub = pp.first;
        oldSub->Ref();
        VLOG(2) << "    Moving subs: sub " << as_hex(oldSub) << " with ticket " << oldTicket;

        auto newSub = oldSub->clone(newRoot);
        for (auto &entry : pp.second) {
            entry->CopyProperties(*firstEntry);

            VLOG(3) << "    Moving sub entry: entry " << as_hex(entry)
                    << " (ref " << as_hex(entry->ref) << ") with ticket {}" << oldTicket;
            // Only need to move first ref entry
            if (entry->ref && movedReferences.count(entry->ref) > 0) {
                continue;
            }

            VLOG(3) << "    Actually Moving sub entry: entry " << as_hex(entry)
                    << " (ref " << as_hex(entry->ref) << ") with ticket " << oldTicket;

            auto t = tf::remote::PagingHelper::cloneWithNewBuffer(*entry->RefOrVal(),
                                                                    newSub);
            if (entry->ref) {
                *entry->ref = std::move(t);
            } else {
                entry->SetVal(std::move(t));
            }
        }
        DCHECK(oldSub->RefCountIsOne());
        oldSub->Unref();

        newSubs[newSub] = std::move(pp.second);
    }
    using std::swap;
    swap(tree.subs, newSubs);

    DCHECK(oldRoot->RefCountIsOne());
    VLOG(2) << "Releasing old root buffer " << as_hex(oldRoot)
            << " with data block at " << as_hex(oldRoot->data()) << " of size " << oldRoot->size();
    oldRoot->Unref();

    return true;
}
