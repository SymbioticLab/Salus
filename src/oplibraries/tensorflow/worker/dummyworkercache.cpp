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

#include "dummyworkercache.h"
#include "platform/logging.h"

namespace tf = ::tensorflow;

namespace salus::oplib::tensorflow {

void EmptyWorkerCache::ListWorkers(std::vector<std::string> *workers) const
{
    DCHECK(workers);
    workers->clear();
}

tf::WorkerInterface *EmptyWorkerCache::CreateWorker(const std::string &)
{
    LOG(ERROR) << "EmptyWorkerCache::CreateWorker called!!";
    return nullptr;
}

bool EmptyWorkerCache::GetDeviceLocalityNonBlocking(const std::string &, tf::DeviceLocality *)
{
    LOG(ERROR) << "EmptyWorkerCache::GetDeviceLocalityNonBlocking called!!";
    return false;
}

void EmptyWorkerCache::GetDeviceLocalityAsync(const std::string &, tf::DeviceLocality *,
                                              tf::StatusCallback done)
{
    LOG(ERROR) << "EmptyWorkerCache::GetDeviceLocalityAsync called!!";
    done(tf::errors::Internal("EmptyWorkerCache::GetDeviceLocalityAsync called!"));
}

SingleWorkerCache::SingleWorkerCache(std::unique_ptr<tf::Worker> &&worker, const std::string &workerName)
    : m_worker(std::move(worker))
    , m_workerName(workerName)
{
    DCHECK(m_worker);
}

SingleWorkerCache::~SingleWorkerCache() = default;

void SingleWorkerCache::ListWorkers(std::vector<std::string> *workers) const
{
    DCHECK(workers);
    workers->clear();
    workers->emplace_back(m_workerName);
}

tf::WorkerInterface *SingleWorkerCache::CreateWorker(const std::string &target)
{
    DCHECK_EQ(target, m_workerName);
    VLOG(2) << "Creating worker " << as_hex(m_worker) << " from target " << target;
    return m_worker.get();
}

void SingleWorkerCache::ReleaseWorker(const std::string &target, tf::WorkerInterface *worker)
{
    DCHECK_EQ(target, m_workerName);
    DCHECK_EQ(worker, m_worker.get());
    // we reuse the worker object, so don't delete it.
}

bool SingleWorkerCache::GetDeviceLocalityNonBlocking(const std::string &, tf::DeviceLocality *dl)
{
    DCHECK(dl);
    dl->set_bus_id(0);
    return true;
}

void SingleWorkerCache::GetDeviceLocalityAsync(const std::string &, tf::DeviceLocality *,
                                              tf::StatusCallback done)
{
    LOG(ERROR) << "SingleWorkerCache::GetDeviceLocalityAsync called!!";
    done(tf::errors::Internal("SingleWorkerCache::GetDeviceLocalityAsync called!"));
}
} // namespace salus::oplib::tensorflow
