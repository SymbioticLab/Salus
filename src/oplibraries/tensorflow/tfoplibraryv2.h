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

#ifndef TFOPLIBRARYV2_H
#define TFOPLIBRARYV2_H

#include "oplibraries/ioplibrary.h"
#include <unordered_map>

namespace executor {
class CustomRequest;
} // namespace executor

namespace salus::oplib::tensorflow {
/**
 * @brief TFOpLibrary that uses TFInstance internally
 */
class TFOpLibraryV2 : public IOpLibrary
{
public:
    TFOpLibraryV2() = default;

    bool initialize() override;
    void uninitialize() override;

    bool accepts(const executor::OpKernelDef &operation) override;

    void onCustom(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                  const executor::CustomRequest &req, DoneCallback cb) override;

    void onRunGraph(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                    const executor::RunGraphRequest &req, DoneCallback cb) override;

    void onRun(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop, const executor::RunRequest &req,
               DoneCallback cb) override;
};

} // namespace salus::oplib::tensorflow

#endif // TFOPLIBRARYV2_H
