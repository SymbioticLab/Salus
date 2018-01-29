/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Aetf <aetf@unlimitedcodeworks.xyz>
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

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "tfsession.h"
#include "oplibraries/tensorflow/tfinstance.h"
#include "oplibraries/tensorflow/worker/dummyworkercache.h"
#include "oplibraries/tensorflow/worker/rendezvousmgr.h"

namespace tf = ::tensorflow;

namespace symbiotic::salus::oplib::tensorflow {

namespace {

auto computePool(tf::Env &env)
{
    static std::unique_ptr<tf::thread::ThreadPool> pool(new tf::thread::ThreadPool(&env, "ZrpcCompute", 4));
    return pool.get();
}

class TFSessionPrivate
{
public:
    TFSessionPrivate(TFInstance &inst);

    ~TFSessionPrivate();

    TFInstance &m_inst;

    tf::MasterEnv m_masterEnv;
    std::unique_ptr<ZrpcMaster> m_master;

    tf::WorkerEnv m_workerEnv;
    std::unique_ptr<ZrpcWorker> m_worker;

    std::unique_ptr<MDGraphMgr> m_graphMgr;
    std::unique_ptr<SalusRendezvousMgr> m_rendezvousMgr;

    tf::ResourceMgr m_resourceMgr;
};

} // namespace

TFSession::TFSession(TFInstance &inst)
    : d(inst)
{
}

TFSession::~TFSession() = default;

TFSessionPrivate::~TFSessionPrivate() {}

TFSessionPrivate::TFSessionPrivate(TFInstance &inst)
    : m_inst(inst)
{
    m_masterEnv.env = &m_inst.env();
    m_workerEnv.env = &m_inst.env();

    // Configure shared devices between master and worker.
    m_masterEnv.local_devices = m_inst.devices();
    m_workerEnv.device_mgr = &m_inst.deviceMgr();
    m_workerEnv.worker_name = m_inst.namePrefix();

    // Create a dummy worker cache, because we don't use remote TF workers.
    auto worker_cache = std::make_unique<DummyWorkerCache>();
    m_masterEnv.worker_cache = worker_cache.get();
    m_masterEnv.worker_cache_factory = DummyWorkerCacheFactory;

    // Finish setting up master environment.
    m_masterEnv.ops = tf::OpRegistry::Global();
    m_masterEnv.master_session_factory = [](auto options, auto env, auto remote_devs, auto worker_cache,
                                            auto device_set) {
        return new tf::MasterSession(options, env, std::move(remote_devs), std::move(worker_cache),
                                     std::move(device_set), tf::CreateNoOpStatsPublisher);
    };

    // Finish setting up worker environment, moving in worker_cache
    m_workerEnv.worker_cache = m_masterEnv.worker_cache;
    m_workerEnv.compute_pool = computePool(m_inst.env());
    m_workerEnv.session_mgr = new tf::SessionMgr(&m_workerEnv, "Salus", std::move(worker_cache),
                                                 [](auto server_def, auto worker_cache) {
                                                     WorkerCacheFactoryOptions options(server_def);
                                                     return DummyWorkerCacheFactory(options, worker_cache);
                                                 });

    m_rendezvousMgr = std::make_unique<SalusRendezvousMgr>(&m_workerEnv);
    m_workerEnv.rendezvous_mgr = m_rendezvousMgr.get();

    m_graphMgr = std::make_unique<MDGraphMgr>(&m_workerEnv);
    m_workerEnv.graph_mgr = m_graphMgr.get();

    // Create master and worker
    m_master = NewZrpcMaster(&m_masterEnv);
    m_worker = NewZrpcWorker(&m_workerEnv);
}

} // namespace symbiotic::salus::oplib::tensorflow
