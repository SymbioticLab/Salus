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

#ifndef SALUS_OPLIB_TENSORFLOW_DUMMYWORKERCACHE_H
#define SALUS_OPLIB_TENSORFLOW_DUMMYWORKERCACHE_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/tfutils.h"
#include <memory>

namespace salus::oplib::tensorflow {

/**
 * @brief An empty WorkerCache because we don't use remote TF workers.
 */
class DummyWorkerCache : public tf::WorkerCacheInterface
{
public:
    ~DummyWorkerCache() override = default;

    void ListWorkers(std::vector<std::string> *workers) const override;

    tf::WorkerInterface *CreateWorker(const std::string &target) override;

    bool GetDeviceLocalityNonBlocking(const std::string &device, tf::DeviceLocality *locality) override;

    void GetDeviceLocalityAsync(const std::string &device, tf::DeviceLocality *locality,
                                tf::StatusCallback done) override;
};

tf::Status DummyWorkerCacheFactory(const ::tensorflow::WorkerCacheFactoryOptions &options,
                                   tf::WorkerCacheInterface **inout);

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_DUMMYWORKERCACHE_H
