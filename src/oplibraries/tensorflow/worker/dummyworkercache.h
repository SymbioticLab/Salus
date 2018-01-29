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

#ifndef SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_DUMMYWORKERCACHE_H
#define SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_DUMMYWORKERCACHE_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include <memory>

namespace symbiotic::salus::oplib::tensorflow {

/**
 * @brief An empty WorkerCache because we don't use remote TF workers.
 */
class DummyWorkerCache : public ::tensorflow::WorkerCacheInterface
{
public:
    ~DummyWorkerCache() override = default;

    void ListWorkers(std::vector<std::string> *workers) const override;

    WorkerInterface *CreateWorker(const std::string &target) override;

    bool GetDeviceLocalityNonBlocking(const std::string &device,
                                      ::tensorflow::DeviceLocality *locality) override;

    void GetDeviceLocalityAsync(const std::string &device, ::tensorflow::DeviceLocality *locality,
                                ::tensorflow::StatusCallback done) override;
};

::tensorflow::Status DummyWorkerCacheFactory(const ::tensorflow::WorkerCacheFactoryOptions &options,
                                             ::tensorflow::WorkerCacheInterface **inout);

} // namespace symbiotic::salus::oplib::tensorflow

#endif // SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_DUMMYWORKERCACHE_H
