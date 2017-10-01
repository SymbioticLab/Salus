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

    INFO("Src dev context {}, dst dev context {}, "
         "source tensor buffer addr: {}, target tensor buffer addr: {}",
         as_hex(entry.device_context), as_hex(dstCtx),
         as_hex(input->tensor_data().data()),
         as_hex(copy.tensor_data().data()));

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
    entry.alloc_ticket = dstDevice->resourceContext().ticket();

    return tf::Status::OK();
}

bool moveTensorTree(const TensorBufferTree &tree, const std::shared_ptr<PerOpAllocDevice> &dstDevice)
{
    assert(!tree.roots.empty());

    auto oldRoot = tf::remote::PagingHelper::bufferOf(*tree.roots[0]->RefOrVal());
    auto oldTicket = tree.roots[0]->alloc_ticket;

    auto oldCount = tf::remote::PagingHelper::refCountOf(*oldRoot);
    DEBUG("    Paging visiting buffer {} (count {}) with ticket {}",
          as_hex(oldRoot), oldCount, oldTicket);

    oldRoot->Ref();

    std::unordered_set<tf::Tensor*> movedReferences;
    Entry *firstEntry = nullptr;
    tf::TensorBuffer *newRoot = nullptr;
    // Firstly page out root buffer
    for (auto entry : tree.roots) {
        if (!newRoot) {
            // only need to actually move the first in roots
            DEBUG("    Actually move first in roots: entry {} (ref {}) with ticket {}",
                  as_hex(entry), as_hex(entry->ref), oldTicket);
            auto ok = moveTensor(*entry, dstDevice, nullptr, {},
                                 tf::strings::StrCat("Paging tensor of ticket ", oldTicket));
            if (!ok.ok()) {
                ERR("Error when paging: {}", ok);
                return false;
            }
            newRoot = tf::remote::PagingHelper::bufferOf(*(entry->val.get()));
            firstEntry = entry;

            if (entry->ref) {
                movedReferences.insert(entry->ref);
            }
            continue;
        }
        DEBUG("    Move other tensors of same root: entry {} (ref {}) with ticket {}",
              as_hex(entry), as_hex(entry->ref), oldTicket);

        entry->CopyProperties(*firstEntry);
        // only one reference entry need to be moved
        if (entry->ref && movedReferences.count(entry->ref) > 0) {
            continue;
        }
        DEBUG("    Move other tensors of same root: ref {} ticket {} not yet moved or this is value",
              as_hex(entry->ref), oldTicket);

        auto t = tf::remote::PagingHelper::cloneWithNewBuffer(*entry->RefOrVal(), newRoot);
        if (entry->ref) {
            *entry->ref = std::move(t);
            movedReferences.insert(entry->ref);
        } else {
            entry->SetVal(std::move(t));
        }
    }

    assert(newRoot);

    // Secondly re-target sub buffers to new root
    for (auto &pp : tree.subs) {
        auto oldSub = pp.first;
        oldSub->Ref();
        DEBUG("    Moving subs: sub {} with ticket {}", as_hex(oldSub), oldTicket);

        auto newSub = oldSub->clone(newRoot);
        for (auto &entry : pp.second) {
            entry->CopyProperties(*firstEntry);

            DEBUG("    Moving sub entry: entry {} (ref {}) with ticket {}",
                  as_hex(entry), as_hex(entry->ref), oldTicket);
            // Only need to move first ref entry
            if (entry->ref && movedReferences.count(entry->ref) > 0) {
                continue;
            }

            DEBUG("    Actually Moving sub entry: entry {} (ref {}) with ticket {}",
                  as_hex(entry), as_hex(entry->ref), oldTicket);

            auto t = tf::remote::PagingHelper::cloneWithNewBuffer(*entry->RefOrVal(),
                                                                    newSub);
            if (entry->ref) {
                *entry->ref = std::move(t);
            } else {
                entry->SetVal(std::move(t));
            }
        }

        assert(oldSub->RefCountIsOne());
        oldSub->Unref();
    }

    assert(oldRoot->RefCountIsOne());
    DEBUG("Releasing old root buffer {} with data block at {} of size {}",
          as_hex(oldRoot), as_hex(oldRoot->data()), oldRoot->size());
    oldRoot->Unref();

    return true;
}
