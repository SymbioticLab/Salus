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

#ifndef TENSORFLOW_HEADERS_H
#define TENSORFLOW_HEADERS_H

#ifndef NDEBUG
#define NDEBUG
#define NEED_UNDEF_NDEBUG
#endif

#include <tensorflow/core/common_runtime/copy_tensor.h>
#include <tensorflow/core/common_runtime/device.h>
#include <tensorflow/core/common_runtime/device_factory.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/common_runtime/dma_helper.h>
#include <tensorflow/core/common_runtime/executor.h>
#include <tensorflow/core/common_runtime/function.h>
#include <tensorflow/core/common_runtime/gpu/gpu_device.h>
#include <tensorflow/core/common_runtime/gpu/gpu_stream_util.h>
#include <tensorflow/core/common_runtime/gpu/process_state.h>
#include <tensorflow/core/common_runtime/graph_optimizer.h>
#include <tensorflow/core/common_runtime/local_device.h>
#include <tensorflow/core/common_runtime/memory_types.h>
#include <tensorflow/core/common_runtime/optimization_registry.h>
#include <tensorflow/core/common_runtime/pending_counts.h>
#include <tensorflow/core/common_runtime/renamed_device.h>
#include <tensorflow/core/common_runtime/shape_refiner.h>
#include <tensorflow/core/common_runtime/step_stats_collector.h>
#include <tensorflow/core/distributed_runtime/base_rendezvous_mgr.h>
#include <tensorflow/core/distributed_runtime/master_env.h>
#include <tensorflow/core/distributed_runtime/master_session.h>
#include <tensorflow/core/distributed_runtime/session_mgr_interface.h>
#include <tensorflow/core/distributed_runtime/worker.h>
#include <tensorflow/core/distributed_runtime/worker_cache.h>
#include <tensorflow/core/distributed_runtime/zrpc/exechelper/paginghelper.h>
#include <tensorflow/core/distributed_runtime/zrpc/exechelper/memorytypes.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/function.h>
#include <tensorflow/core/framework/function.pb.h>
#include <tensorflow/core/framework/graph_def_util.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_segment.h>
#include <tensorflow/core/framework/rendezvous.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/graph/algorithm.h>
#include <tensorflow/core/graph/graph.h>
#include <tensorflow/core/graph/graph_constructor.h>
#include <tensorflow/core/graph/graph_partition.h>
#include <tensorflow/core/graph/validate.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/gtl/flatmap.h>
#include <tensorflow/core/lib/gtl/flatset.h>
#include <tensorflow/core/lib/gtl/inlined_vector.h>
#include <tensorflow/core/lib/gtl/manual_constructor.h>
#include <tensorflow/core/lib/gtl/stl_util.h>
#include <tensorflow/core/lib/strings/strcat.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/mutex.h>
#include <tensorflow/core/protobuf/config.pb.h>
#include <tensorflow/core/protobuf/master.pb.h>
#include <tensorflow/core/protobuf/worker.pb.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/util/tensor_slice_reader_cache.h>
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>

#ifdef NEED_UNDEF_NDEBUG
#undef NDEBUG
#undef NEED_UNDEF_NDEBUG
#endif

#include "oplibraries/tensorflow/tfutils.h"

// Include our logging to override TF's logging, and to provide stream operators
#include "platform/logging.h"

inline std::ostream &operator<<(std::ostream &os, const tensorflow::AllocatorAttributes &c)
{
    return os << "tensorflow::AllocatorAttributes("
              << "on_host=" << c.on_host() << ", nic_compatible=" << c.nic_compatible()
              << ", gpu_compatible=" << c.gpu_compatible() << ")";
}

inline std::ostream &operator<<(std::ostream &os, const tensorflow::AllocationAttributes &c)
{
    return os << "tensorflow::AllocationAttributes("
              << "allocation_will_be_logged=" << c.allocation_will_be_logged
              << ", no_retry_on_failure=" << c.no_retry_on_failure << ")";
}

inline std::ostream &operator<<(std::ostream &os, const tensorflow::TensorValue &c)
{
    os << "TensorValue(";
    if (c.tensor) {
        os << "shape: " << c->shape().DebugString() << ", datatype: " << c->dtype()
           << ", is ref: " << c.is_ref();
        if (c.is_ref()) {
            os << " (mutex: " << as_hex(c.mutex_if_ref) << ")";
        }
        os << ", buffer: " << as_hex(c->tensor_data().data());
    } else {
        os << "<empty>";
    }
    os << ")";
    return os;
}

#endif // TENSORFLOW_HEADERS_H
