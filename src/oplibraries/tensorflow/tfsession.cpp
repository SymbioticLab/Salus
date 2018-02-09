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
#include "oplibraries/tensorflow/worker/mdgraphmgr.h"

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
    TFSessionPrivate(TFInstance &inst, ExecutionContext &&ctx, const tf::ConfigProto &config, tf::GraphDef *gdef);

    ~TFSessionPrivate();

#define DECLARE_HANDLER_PRIV(name) \
    void handle ## name (P ## name ## Request &&req, name ## Callback cb)

    DECLARE_HANDLER_PRIV(ExtendSession);
    DECLARE_HANDLER_PRIV(PartialRunSetup);
    DECLARE_HANDLER_PRIV(RunStep);

#undef DECLARE_HANDLER_PRIV

    std::string handle() const;

    void safeClose(std::shared_ptr<TFSession> &&self);

    TFInstance &m_inst;

    tf::MasterEnv m_masterEnv;
    utils::ScopedUnref<tf::MasterSession> m_masterSess;

    tf::WorkerEnv m_workerEnv;
    std::unique_ptr<tf::Worker> m_worker;

    std::unique_ptr<MDGraphMgr> m_graphMgr;
    std::unique_ptr<SalusRendezvousMgr> m_rendezvousMgr;

    tf::ResourceMgr m_resourceMgr;

    ExecutionContext m_execCtx;
};

} // namespace

TFSession::TFSession(TFInstance &inst, ExecutionContext &&ctx, const tf::ConfigProto &config, tf::GraphDef *gdef);
    : d(inst, std::move(ctx), config, gdef)
{
}

TFSession::~TFSession() = default;

std::string TFSession::handle() const
{
    return d->handle();
}

void TFSession::safeClose()
{
    return d->safeClose(shared_from_this());
}

#define IMPL_HANDLER(name) \
void TFSession::handle ## name (P ## name ## Request &&req, name ## Callback cb) \
{ \
    d->handle ## name(std::move(req), std::move(cb)); \
}

    IMPL_HANDLER(ExtendSession)
    IMPL_HANDLER(PartialRunSetup)
    IMPL_HANDLER(RunStep)

#undef IMPL_HANDLER

TFSessionPrivate::~TFSessionPrivate() {}

std::string TFSessionPrivate::handle() const
{
    DCHECK(m_masterSess);
    return m_masterSess->handle();
}

TFSessionPrivate::TFSessionPrivate(TFInstance &inst, ExecutionContext &&ctx, const tf::ConfigProto &config, tf::GraphDef *gdef)
    : m_inst(inst)
{
    // Populate master and worker env
    // FIXME: move this into its own function
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
    m_masterEnv.master_session_factory = nullptr;

    // Finish setting up worker environment, moving in worker_cache
    m_workerEnv.worker_cache = m_masterEnv.worker_cache;
    m_workerEnv.compute_pool = computePool(m_inst.env());
    m_workerEnv.session_mgr = new tf::SessionMgr(&m_workerEnv, "Salus", std::move(worker_cache),
                                                 [](auto server_def, auto worker_cache) {
                                                     tf::WorkerCacheFactoryOptions options(server_def);
                                                     return DummyWorkerCacheFactory(options, worker_cache);
                                                 });

    m_rendezvousMgr = std::make_unique<SalusRendezvousMgr>(&m_workerEnv);
    m_workerEnv.rendezvous_mgr = m_rendezvousMgr.get();

    m_graphMgr = std::make_unique<MDGraphMgr>(&m_workerEnv);
    m_workerEnv.graph_mgr = m_graphMgr.get();

    // Create worker
    m_worker = std::make_unique<tf::Worker>(&m_workerEnv);

    auto device_set = std::make_unique<tf::DeviceSet>();
    int num_local_devices = 0;
    for (auto d : m_masterEnv.local_devices) {
        device_set->AddDevice(d);
        if (num_local_devices == 0) {
            // Uses the first local device as the client device.
            device_set->set_client_device(d);
        }
        num_local_devices++;
    }
    DCHECK(device_set->client_device()) << "No client device found. Missing " << "CPU:0 device?";

    tf::SessionOptions options;
    options.config = config;
    m_masterSess = utils::wrap_unref(new tf::MasterSession(options, &m_masterEnv,
                                                           std::make_unique<std::vector<std::unique_ptr<tf::Device>>>(),
                                                           nullptr, std::move(device_set), tf::CreateNoOpStatsPublisher));

    status = m_masterSess->Create(gdef, {});
    if (!status.ok()) {
        m_masterSess->Close().IgnoreError();
        throw TFException(status);
    }

    m_execCtx.acceptOffer(handle());
    // Only take passed in ctx after we are sure to succeed
    m_execCtx = std::move(ctx);
}

void TFSessionPrivate::safeClose(std::shared_ptr<TFSession> &&self)
{
    DCHECK(m_masterSess);
    SALUS_THROW_IF_ERROR(m_masterSess->Close());
    m_execCtx.deleteSession([self = std::move(self)](){
        // Retains alive until execution engine actually deletes its internal resources.
    });
}

void TFSession::handleExtendSession(ZmqServer::Sender sender, const tf::ExtendSessionRequest &req, tf::ExtendSessionResponse &resp, StatusCallback &&cb)
{
    // FIXME
}

void TFSession::handlePartialRunSetup(ZmqServer::Sender sender, const tf::PartialRunSetupRequest &req, tf::PartialRunSetupResponse &resp, StatusCallback &&cb)
{
    
    // FIXME
}

void TFSession::handleRunSetup(ZmqServer::Sender sender, const tf::RunSetupRequest &req, tf::RunSetupResponse &resp, StatusCallback &&cb)
{
    
    // FIXME
}
} // namespace symbiotic::salus::oplib::tensorflow
