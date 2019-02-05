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

#ifndef SALUS_OPLIB_TENSORFLOW_DUMMYWORKERCACHE_H
#define SALUS_OPLIB_TENSORFLOW_DUMMYWORKERCACHE_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/tfutils.h"
#include <memory>

namespace salus::oplib::tensorflow {

/**
 * @brief An empty WorkerCache because we don't use remote TF workers.
 */
class EmptyWorkerCache : public tf::WorkerCacheInterface
{
public:
    ~EmptyWorkerCache() override = default;

    void ListWorkers(std::vector<std::string> *workers) const override;

    tf::WorkerInterface *CreateWorker(const std::string &target) override;

    bool GetDeviceLocalityNonBlocking(const std::string &device, tf::DeviceLocality *locality) override;

    void GetDeviceLocalityAsync(const std::string &device, tf::DeviceLocality *locality,
                                tf::StatusCallback done) override;
};

class SingleWorkerCache : public tf::WorkerCacheInterface
{
    std::unique_ptr<tf::Worker> m_worker;
    std::string m_workerName;

public:
    explicit SingleWorkerCache(std::unique_ptr<tf::Worker> &&worker, const std::string &workerName);
    ~SingleWorkerCache() override;

    void ListWorkers(std::vector<std::string> *workers) const override;

    tf::WorkerInterface *CreateWorker(const std::string &target) override;
    void ReleaseWorker(const std::string& target, tf::WorkerInterface* worker) override;

    bool GetDeviceLocalityNonBlocking(const std::string &device, tf::DeviceLocality *locality) override;

    void GetDeviceLocalityAsync(const std::string &device, tf::DeviceLocality *locality,
                                tf::StatusCallback done) override;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_DUMMYWORKERCACHE_H
