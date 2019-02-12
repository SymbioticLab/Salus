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

#ifndef SALUS_OPLIB_TENSORFLOW_TFSESSION_H
#define SALUS_OPLIB_TENSORFLOW_TFSESSION_H

#include "oplibraries/tensorflow/tfoplibraryv2.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "utils/macros.h"
#include "utils/pointerutils.h"

#include <memory>

namespace tensorflow {
class Device;
} // namespace tensorflow

namespace salus {
class ExecutionContext;
namespace oplib::tensorflow {
class TFInstance;
struct HandlerCallback;

/**
 * @brief One session of job
 */
class TFSession : public std::enable_shared_from_this<TFSession>
{
public:
    SALUS_DISALLOW_COPY_AND_ASSIGN(TFSession);

    TFSession(TFInstance &inst, std::shared_ptr<ExecutionContext> ctx, std::vector<tf::Device *> devices,
              const tf::ConfigProto &config,
              tf::GraphDef *gdef);

    ~TFSession();

    std::string handle() const;

    /**
     * @brief Safely close the session.
     *
     * The resource is deleted later when execution engine actually removes internal structure.
     */
    void safeClose();

    /**
     * @brief Defer close after the removal of execCtx
     */
    void deferClose(HandlerCallback &&cb);

#define DECLARE_HANDLER(name)                                                                                \
    void handle##name(const tf::name##Request &req, tf::name##Response &resp, HandlerCallback &&cb)

    DECLARE_HANDLER(ExtendSession);

    DECLARE_HANDLER(PartialRunSetup);

    DECLARE_HANDLER(RunStep);

#undef DECLARE_HANDLER

private:
    class TFSessionPrivate;

    sstl::PImpl<TFSessionPrivate> d;
};

} // namespace oplib::tensorflow
} // namespace salus

#endif // SALUS_OPLIB_TENSORFLOW_TFSESSION_H
