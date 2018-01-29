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

namespace symbiotic::salus::oplib::tensorflow {

tf::Status DummyWorkerCacheFactory(const tf::WorkerCacheFactoryOptions &, tf::WorkerCacheInterface **inout)
{
    *inout = new DummyWorkerCache();
    return tf::Status::OK();
}

void DummyWorkerCache::ListWorkers(std::vector<std::string> *workers) const
{
    DCHECK(workers);
    workers->clear();
}

tf::WorkerInterface *DummyWorkerCache::CreateWorker(const std::string &target)
{
    LOG(ERROR) << "DummyWorkerCache::CreateWorker called!!";
    return nullptr;
}

bool DummyWorkerCache::GetDeviceLocalityNonBlocking(const std::string &, tf::DeviceLocality *)
{
    LOG(ERROR) << "DummyWorkerCache::GetDeviceLocalityNonBlocking called!!";
    return false;
}

void DummyWorkerCache::GetDeviceLocalityAsync(const std::string &, tf::DeviceLocality *,
                                              tf::StatusCallback done)
{
    LOG(ERROR) << "DummyWorkerCache::GetDeviceLocalityAsync called!!";
    done(tf::errors::Internal("DummyWorkerCache::GetDeviceLocalityAsync called!"));
}

} // namespace symbiotic::salus::oplib::tensorflow
