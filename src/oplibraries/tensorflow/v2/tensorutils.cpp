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
        // Remember to destory val first
        entry.ClearVal();
        entry.SetVal(std::move(copy));
    }

    entry.alloc_attr = attr;
    entry.device_context = dstCtx;
    entry.device = dstDevice;
    entry.alloc_ticket = dstDevice->resourceContext().ticket;

    return tf::Status::OK();
}
