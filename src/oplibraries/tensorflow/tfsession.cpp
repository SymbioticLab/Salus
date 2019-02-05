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

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "oplibraries/tensorflow/tfsession.h"

#include "execution/executionengine.h"
#include "oplibraries/tensorflow/handlercallback.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/tfinstance.h"
#include "oplibraries/tensorflow/v3/sigraphmgr.h"
#include "oplibraries/tensorflow/worker/dummysessionmgr.h"
#include "oplibraries/tensorflow/worker/dummyworkercache.h"
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
    TFSessionPrivate(TFInstance &inst, std::shared_ptr<ExecutionContext> &&ctx, std::vector<tf::Device *> devices,
                     const tf::ConfigProto &config, tf::GraphDef *gdef);

    ~TFSessionPrivate();

#define DECLARE_HANDLER_PRIV(name)                                                                                     \
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

TFSession::TFSession(TFInstance &inst, std::shared_ptr<ExecutionContext> ctx, std::vector<tf::Device *> devices,
                     const tf::ConfigProto &config,
                     tf::GraphDef *gdef)
    : d(std::make_unique<TFSessionPrivate>(inst, std::move(ctx), std::move(devices), config, gdef))
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

#define IMPL_HANDLER(name)                                                                                             \
    void TFSession::handle##name(const tf::name##Request &req, tf::name##Response &resp, HandlerCallback &&cb)         \
    {                                                                                                                  \
        d->handle##name(req, resp, std::move(cb));                                                                     \
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
                                              std::vector<tf::Device *> devices, const tf::ConfigProto &config, tf::GraphDef *gdef)
    : m_inst(inst)
    , m_sessMgr(nullptr)
    , m_rendezvousMgr(nullptr)
{
    // Populate worker env first
    m_workerEnv.env = &m_inst.env();
    m_workerEnv.compute_pool = computePool(m_inst.env());
    m_workerEnv.local_devices = std::move(devices);

    // No one is using workerEnv's device_mgr
    m_workerEnv.device_mgr = nullptr;

    m_rendezvousMgr = std::make_unique<SalusRendezvousMgr>(&m_workerEnv);
    m_workerEnv.rendezvous_mgr = m_rendezvousMgr.get();

    // create and set session_mgr to worker_env, as the last step
    // NOTE that session_mgr also needs a worker_env to create session later on,
    // but not at this point, so it's ok to pass in a partially configured worker_env
    // Setup session manager that creates worker session later
    m_sessMgr = std::make_unique<LocalSessionMgr>(
        GetCreateWorkerSessionFnForSIGraphMgr(m_inst.namePrefix(), &m_workerEnv, ctx, config));
    m_workerEnv.session_mgr = m_sessMgr.get();

    // Create a worker using the worker_env and add it to the cache, containing this only local worker
    auto workerCache =
        std::make_unique<SingleWorkerCache>(std::make_unique<tf::Worker>(&m_workerEnv), m_inst.namePrefix());

    // Populate master env
    m_masterEnv.env = &m_inst.env();
    m_masterEnv.ops = tf::OpRegistry::Global();
    // MasterSession don't use local_devices when it's given workerCache, making it empty to make sure
    m_masterEnv.local_devices.clear();
    auto device_set = std::make_unique<tf::DeviceSet>();
    for (auto d : m_workerEnv.local_devices) {
        device_set->AddDevice(d);
    }
    // Uses the first local device as the client device.
    DCHECK(!m_workerEnv.local_devices.empty()) << "No client device found. Missing CPU:0 device?";
    device_set->set_client_device(m_workerEnv.local_devices.front());

    tf::SessionOptions options;
    options.config = config;
    options.config.set_isolate_session_state(true);
    m_masterSess =
        sstl::make_scoped_unref<tf::MasterSession>(options, &m_masterEnv,
                                                   // MasterSession don't use remote_devices when given workerCache
                                                   std::make_unique<std::vector<std::unique_ptr<tf::Device>>>(),
                                                   std::move(workerCache), std::move(device_set),
                                                   tf::CreateNoOpStatsPublisher);

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
                                                        tf::PartialRunSetupResponse &resp, HandlerCallback &&cb)
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
    LOG(INFO) << "Defer closing session " << d->handle();

    d->m_execCtx->finish([self = shared_from_this(), cb = std::move(cb.cb), raw_tfresp]() mutable {
        HandlerCallback hcb;
        hcb.tfresp = sstl::wrap_unique(raw_tfresp);
        hcb.cb = std::move(cb);

        self->safeClose();
        hcb(Status::OK());
    });
}

} // namespace salus::oplib::tensorflow
