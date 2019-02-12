/*
 * Copyright 2019 Peifeng Yu <peifeng@umich.edu>
 * 
 * This file is part of Salus
 * (see https://github.com/SymbioticLab/Salus).
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SALUS_OPLIB_TENSORFLOW_SIGRAPHMGR_H
#define SALUS_OPLIB_TENSORFLOW_SIGRAPHMGR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "execution/executionengine.h"
#include "oplibraries/tensorflow/worker/dummysessionmgr.h"
#include "resources/resources.h"
#include "utils/macros.h"

namespace salus::oplib::tensorflow {

/**
 * @brief A simple graph manager that doesn't hook into anything,
 * and only controls when an iteration starts
 */
class SIGraphMgr : public tf::GraphMgr
{
public:
    explicit SIGraphMgr(const tf::WorkerEnv *env, tf::DeviceMgr *deviceMgr, std::shared_ptr<ExecutionContext> execCtx);
    ~SIGraphMgr() override;

    Status Register(const std::string &session, const tf::GraphDef &gdef,
                    const tf::GraphOptions &graph_options, const tf::DebugOptions &debug_options,
                    tf::DistributedFunctionLibraryRuntime *cluster_flr, std::string *handle) override;

protected:
    tf::Status InitSIItem(const std::string &session, const tf::GraphDef &gdef,
                          const tf::GraphOptions &graph_options, const tf::DebugOptions &debug_options,
                          tf::DistributedFunctionLibraryRuntime *cluster_flr, Item &item);

private:
    std::shared_ptr<ExecutionContext> m_execCtx;
};

CreateWorkerSessionFn GetCreateWorkerSessionFnForSIGraphMgr(const std::string &worker_name,
                                                            const tf::WorkerEnv *worker_env,
                                                            std::shared_ptr<ExecutionContext> execCtx,
                                                            const tf::ConfigProto &config);

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_SIGRAPHMGR_H
