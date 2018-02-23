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
