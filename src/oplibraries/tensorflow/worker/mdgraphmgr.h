/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Peifeng Yu <peifeng@umich.edu>
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

#ifndef SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_MDGRAPHMGR_H
#define SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_MDGRAPHMGR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "utils/macros.h"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace symbiotic::salus::oplib::tensorflow {

/**
 * @brief MDGraphMgr keeps track of graph execution.
 *
 * Each computation graph is registered with MDGraphMgr. Then it is executed through it with multiple step_id.
 * This MDGraphMgr creates executor that is compatible with Salus internal execution engine.
 */
class MDGraphMgr : public tf::GraphMgr
{
public:
    explicit MDGraphMgr(const tf::WorkerEnv *env, tf::DeviceMgr *device_mgr);
    ~MDGraphMgr() override;

protected:
    tf::Status InitItem(const std::string &session, const ::tensorflow::GraphDef &gdef,
                        const tf::GraphOptions &graph_options, const tf::DebugOptions &debug_options,
                        tf::DistributedFunctionLibraryRuntime *cluster_flr, Item *item) override;

private:
    // Global Resource manager shared by all local devices.
    // NOTE: Must be valid when destorying opsegment, because
    // some op uses this during deconstruction.
    std::unique_ptr<tf::ResourceMgr> m_resourceMgr;

    // Global Opsegment shared by all local devices on all workers
    // (we have one and only one local worker)
    std::unique_ptr<tf::OpSegment> m_opseg;

    // Kernel to device name map
    std::unordered_map<const tf::OpKernel *, std::string> m_kernelToDevice;
    std::mutex m_mu;
};

} // namespace symbiotic::salus::oplib::tensorflow

#endif // SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_MDGRAPHMGR_H
