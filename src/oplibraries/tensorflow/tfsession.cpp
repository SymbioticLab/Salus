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

#include "tfsession.h"

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/tfinstance.h"

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

    std::unique_ptr<MDGraphMgr> m_graphMgr;
    tf::WorkerEnv m_workerEnv;
    std::unique_ptr<ZrpcWorker> m_worker;

    tf::ResourceMgr m_resourceMgr;
};

} // namespace

TFSession::TFSession(TFInstance &inst)
    : d(std::make_unique<TFSessionPrivate>(inst))
{
}

TFSession::~TFSession() = default;

TFSessionPrivate::~TFSessionPrivate()
{
}

TFSessionPrivate::TFSessionPrivate(TFInstance &inst)
    : m_inst(inst)
{
    m_masterEnv.env = &m_inst.env();
    m_workerEnv.env = &m_inst.env();

    // Configure shared devices between master and worker.
    m_masterEnv.local_devices = m_inst.devices();
    m_workerEnv.device_mgr = &m_inst.deviceMgr();
    m_workerEnv.worker_name = m_inst.namePrefix();

    // Finish setting up master environment.
    m_masterEnv.ops = tf::OpRegistry::Global();
    m_masterEnv.master_session_factory =
        [](tf::SessionOptions options, const tf::MasterEnv *env,
                 std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
                 std::unique_ptr<WorkerCacheInterface> worker_cache,
                std::unique_ptr<DeviceSet> device_set) {
            return new tf::MasterSession(options, env, std::move(remote_devs), std::move(worker_cache),
                                     std::move(device_set), tf::CreateNoOpStatsPublisher);
        };

    // Finish setting up worker environment.
    m_workerEnv.compute_pool = computePool(m_inst.env());
    m_workerEnv.rendezvous_mgr = new ZrpcRendezvousMgr(&m_workerEnv);
    m_workerEnv.session_mgr =
        new SessionMgr(&m_workerEnv, "Salus", std::unique_ptr<WorkerCacheInterface>(worker_cache),
                       [](const ServerDef &server_def, WorkerCacheInterface **worker_cache) {
                           WorkerCacheFactoryOptions options(server_def);
                           return WorkerCacheFactory(options, worker_cache);
                       });

    m_graphMgr = std::make_unique<MDGraphMgr>(&m_workerEnv);
    m_workerEnv.graph_mgr = m_graphMgr.get();

    // Create master and worker
    m_master = NewZrpcMaster(&m_masterEnv);
    m_worker = NewZrpcWorker(&m_workerEnv);
}

} // namespace symbiotic::salus::oplib::tensorflow
