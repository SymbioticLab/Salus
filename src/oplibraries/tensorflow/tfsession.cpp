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
#include "oplibraries/tensorflow/tfsession.h"
#include "execution/executionengine.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/tfinstance.h"
#include "oplibraries/tensorflow/worker/dummysessionmgr.h"
#include "oplibraries/tensorflow/worker/dummyworkercache.h"
#include "oplibraries/tensorflow/worker/mdgraphmgr.h"
#include "oplibraries/tensorflow/worker/rendezvousmgr.h"

#include <cmath>

namespace salus::oplib::tensorflow {

namespace {

auto computePool(tf::Env &env)
{
    static std::unique_ptr<tf::thread::ThreadPool> pool(new tf::thread::ThreadPool(&env, "ZrpcCompute", 4));
    return pool.get();
}

} // namespace

class TFSession::TFSessionPrivate
{
public:
    TFSessionPrivate(TFInstance &inst, std::shared_ptr<ExecutionContext> &&ctx, const tf::ConfigProto &config,
                     tf::GraphDef *gdef);

    ~TFSessionPrivate();

#define DECLARE_HANDLER_PRIV(name)                                                                           \
    void handle##name(const tf::name##Request &req, tf::name##Response &resp, HandlerCallback &&cb)

    DECLARE_HANDLER_PRIV(ExtendSession);
    DECLARE_HANDLER_PRIV(PartialRunSetup);
    DECLARE_HANDLER_PRIV(RunStep);

#undef DECLARE_HANDLER_PRIV

    std::string handle() const;

    void safeClose(std::shared_ptr<TFSession> &&self);

    TFInstance &m_inst;

    // execution context must be the last to be destroied.
    // which also removes the session item from execution engine.
    std::shared_ptr<ExecutionContext> m_execCtx;

    tf::WorkerEnv m_workerEnv;
    tf::MasterEnv m_masterEnv;

    // master session -> worker cache -> worker
    sstl::ScopedUnref<tf::MasterSession> m_masterSess;

    // owns graph mgr
    std::unique_ptr<LocalSessionMgr> m_sessMgr;

    std::unique_ptr<SalusRendezvousMgr> m_rendezvousMgr;
};

TFSession::TFSession(TFInstance &inst, std::shared_ptr<ExecutionContext> ctx, const tf::ConfigProto &config,
                     tf::GraphDef *gdef)
    : d(std::make_unique<TFSessionPrivate>(inst, std::move(ctx), config, gdef))
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

#define IMPL_HANDLER(name)                                                                                   \
    void TFSession::handle##name(const tf::name##Request &req, tf::name##Response &resp,                     \
                                 HandlerCallback &&cb)                                                       \
    {                                                                                                        \
        d->handle##name(req, resp, std::move(cb));                                                           \
    }

IMPL_HANDLER(ExtendSession)
IMPL_HANDLER(PartialRunSetup)
IMPL_HANDLER(RunStep)

#undef IMPL_HANDLER

TFSession::TFSessionPrivate::~TFSessionPrivate() = default;

std::string TFSession::TFSessionPrivate::handle() const
{
    DCHECK(m_masterSess);
    return m_masterSess->handle();
}

TFSession::TFSessionPrivate::TFSessionPrivate(TFInstance &inst, std::shared_ptr<ExecutionContext> &&ctx,
                                              const tf::ConfigProto &config, tf::GraphDef *gdef)
    : m_inst(inst)
    , m_sessMgr(nullptr)
    , m_rendezvousMgr(nullptr)
{
    // Get resource estimation from client
    ResStats rm{};
    ResourceTag rt = resources::GPU0Memory;
    auto &m = config.salus_options().resource_map();
    if (auto oval = sstl::optionalGet(m.persistant(), rt.DebugString())) {
        rm.persist = static_cast<size_t>(std::round(*oval));
    }
    if (auto oval = sstl::optionalGet(m.temporary(), rt.DebugString())) {
        rm.temporary = static_cast<size_t>(std::round(*oval));
    }
    rm.count = 0;

    // Setup session manager that creates worker session later
    m_sessMgr = std::make_unique<LocalSessionMgr>([this, ctx, rm](auto sessHandle) {
        // worker session takes ownership of a deviceMgr, so we create shadow devices for it.
        return std::make_unique<tf::WorkerSession>(sessHandle, m_inst.namePrefix(),
                                                   std::make_unique<EmptyWorkerCache>(),
                                                   // use an empty device mgr for session, because we should use
                                                   // the device mgr from worker_env to make sure we use ISalusDevice
                                                   std::make_unique<tf::DeviceMgr>(std::vector<tf::Device*>{}),
                                                   std::make_unique<MDGraphMgr>(&m_workerEnv, ctx, rm));
    });

    // Populate worker env, which uses the session manager
    m_workerEnv.env = &m_inst.env();
    m_workerEnv.device_mgr = &m_inst.deviceMgr();
    m_workerEnv.compute_pool = computePool(m_inst.env());
    m_workerEnv.session_mgr = m_sessMgr.get();
    m_rendezvousMgr = std::make_unique<SalusRendezvousMgr>(&m_workerEnv);
    m_workerEnv.rendezvous_mgr = m_rendezvousMgr.get();

    // Create a worker cache, containing the only local worker
    auto workerCache = std::make_unique<SingleWorkerCache>(std::make_unique<tf::Worker>(&m_workerEnv), m_inst.namePrefix());

    // Populate master env
    m_masterEnv.env = &m_inst.env();
    m_masterEnv.local_devices = m_inst.devices();
    m_masterEnv.ops = tf::OpRegistry::Global();

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
    DCHECK(device_set->client_device()) << "No client device found. Missing CPU:0 device?";

    tf::SessionOptions options;
    options.config = config;
    options.config.set_isolate_session_state(true);
    m_masterSess = sstl::make_scoped_unref<tf::MasterSession>(
        options, &m_masterEnv, std::make_unique<std::vector<std::unique_ptr<tf::Device>>>(),
        std::move(workerCache), std::move(device_set), tf::CreateNoOpStatsPublisher);

    // Call create on master session to finalize
    auto status = m_masterSess->Create(gdef, {});
    if (!status.ok()) {
        m_masterSess->Close().IgnoreError();
        throw TFException(status);
    }

    // Only take passed in ctx after we are sure to succeed
    m_execCtx = std::move(ctx);
    m_execCtx->setSessionHandle(handle());
}

void TFSession::TFSessionPrivate::safeClose(std::shared_ptr<TFSession> &&self)
{
    DCHECK(m_masterSess);
    DCHECK(self);
    LOG(INFO) << "Closing session " << handle();

    SALUS_THROW_IF_ERROR(m_masterSess->Close());
}

void TFSession::TFSessionPrivate::handleExtendSession(const tf::ExtendSessionRequest &req,
                                                      tf::ExtendSessionResponse &resp, HandlerCallback &&cb)
{
    SALUS_THROW_IF_ERROR(tf::ValidateExternalGraphDefSyntax(req.graph_def()));
    SALUS_THROW_IF_ERROR(m_masterSess->Extend(&req, &resp));
    cb(Status::OK());
}

void TFSession::TFSessionPrivate::handlePartialRunSetup(const tf::PartialRunSetupRequest &req,
                                                        tf::PartialRunSetupResponse &resp,
                                                        HandlerCallback &&cb)
{
    SALUS_THROW_IF_ERROR(m_masterSess->PartialRunSetup(&req, &resp));
    cb(Status::OK());
}

void TFSession::TFSessionPrivate::handleRunStep(const tf::RunStepRequest &req, tf::RunStepResponse &resp,
                                                HandlerCallback &&cb)
{
    tf::CallOptions opts;
    tf::ProtoRunStepRequest wreq(&req);
    tf::NonOwnedProtoRunStepResponse wresp(&resp);
    SALUS_THROW_IF_ERROR(m_masterSess->Run(&opts, wreq, &wresp));
    cb(Status::OK());
}

void TFSession::deferClose(HandlerCallback &&cb)
{
    // cb is move-only, can't be captured and pass to std::function.
    // so we extract and reconstruct inside the lambda
    auto raw_tfresp = cb.tfresp.release();

    d->m_execCtx->finish([self = shared_from_this(), cb = std::move(cb.cb), raw_tfresp]() mutable {
        HandlerCallback hcb;
        hcb.tfresp = sstl::wrap_unique(raw_tfresp);
        hcb.cb = std::move(cb);

        self->safeClose();
        hcb(Status::OK());
    });
}

} // namespace salus::oplib::tensorflow
