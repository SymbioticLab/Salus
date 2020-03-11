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

#include "oplibraries/tensorflow/tfinstance.h"

#include "execution/executionengine.h"
#include "oplibraries/tensorflow/device/gpu/gpu.h"
#include "oplibraries/tensorflow/device/gpu/lane/lanemgr.h"
#include "oplibraries/tensorflow/device/salusdevices.h"
#include "oplibraries/tensorflow/device/shadowdevices.h"
#include "oplibraries/tensorflow/handlercallback.h"
#include "oplibraries/tensorflow/tfexception.h"
#include "oplibraries/tensorflow/tfsession.h"
#include "utils/macros.h"

namespace salus::oplib::tensorflow {

inline tf::StringPiece svToStringPiece(std::string_view sv)
{
    return {sv.data(), sv.size()};
}

/* static */ TFInstance &TFInstance::instance()
{
    static TFInstance inst;
    return inst;
}

TFInstance::TFInstance()
    : m_env(tf::Env::Default())
    , m_devCon()
    , m_laneMgr(std::make_unique<LaneMgr>())
{
}

TFInstance::DeviceContainer::DeviceContainer()
{
    tf::SessionOptions sess_opts;
    // Disable old style RPCDevice creation if any.
    (*sess_opts.config.mutable_device_count())["RPC"] = 0;
    // We don't create GPU device through DeviceFactory
    (*sess_opts.config.mutable_device_count())["GPU"] = 0;

    // Load devices
    maybeRegisterSalusDeviceFactories();
    SALUS_THROW_IF_ERROR(tf::DeviceFactory::AddDevices(sess_opts, namePrefix(), &devices));
    deviceMgr = std::make_unique<tf::DeviceMgr>(devices);

    for (int dt = 0; dt != MaxDeviceType; ++dt) {
        for (int id = 0; id != MaxDeviceId; ++id) {
            deviceMgr->LookupDevice(svToStringPiece(SpecToTFDevName(DeviceSpec{static_cast<DeviceType>(dt), id})),
                                    &specToTF[dt][id]);
        }
    }
}

TFInstance::~TFInstance() = default;

void TFInstance::handleCreateSession(std::unique_ptr<tf::CreateSessionRequest> &&req, tf::CreateSessionResponse &resp,
                                     HandlerCallback &&cb)
{
    SALUS_THROW_IF_ERROR(ValidateExternalGraphDefSyntax(req->graph_def()));

    // NOTE: it's safe to capture resp by reference, because it is actually backed by cb
    auto ectx = ExecutionEngine::instance().makeContext();
    if (!ectx) {
        cb(tf::errors::Aborted("Backend engine interrupted"));
        return;
    }

    // We don't need exclusive mode anymore.
    ectx->dropExlusiveMode();

    LaneMgr::Layout layout;
    // Get resource estimation from client
    constexpr const char *rt[] = {
        "MEMORY:GPU0",
        "MEMORY:GPU1",
        "MEMORY:GPU2",
        "MEMORY:GPU3",
        "MEMORY:GPU4",
        nullptr,
    };
    auto &m = req->config().salus_options().resource_map();
    for (auto iGpu = 0_sz; iGpu != m_laneMgr->numGPUs(); ++iGpu) {
        const auto totalGPUMemory = m_laneMgr->totalMemoryForGPU(iGpu);

        CHECK_NOTNULL(rt[iGpu]) << "We need more GPU strings";

        size_t limit = 0;
        size_t persistant = 0;
        auto p = sstl::optionalGet(m.persistant(), rt[iGpu]);
        auto t = sstl::optionalGet(m.temporary(), rt[iGpu]);
        if (!p || !t) {
            break;
        }
        persistant = static_cast<size_t>(std::round(*p));
        // HACK: scale persistent up 10% to mitigate OOM and fragmentation
        persistant = static_cast<size_t>(persistant * 1.1);
        limit += persistant;

        limit += static_cast<size_t>(std::round(*t));

        // HACK: scale the total up 5%, just to be safe
        limit = static_cast<size_t>(limit * 1.05); // and even more 10%
        limit = std::min(limit, totalGPUMemory); // cap to max value

        layout.memoryLimits.push_back(limit);
        layout.persistentOccupation.push_back(persistant);
    }

    if (layout.memoryLimits.empty()) {
        auto limit = m_laneMgr->totalMemoryForGPU(0);
        layout.memoryLimits.push_back(limit);
        layout.persistentOccupation.push_back(limit);
        LOG(WARNING) << "No resource info for current session, assuming whole GPU allocation: " << limit;
    }

    CHECK_EQ(layout.memoryLimits.size(), layout.persistentOccupation.size());

    auto totalRunningTime =
        static_cast<uint64_t>(std::round(sstl::getOrDefault(m.persistant(), "TIME:TOTAL", 0.0))) * 1000;
    ectx->setExpectedRunningTime(totalRunningTime);

    // smaller is higher priority
    auto priority = static_cast<int>(sstl::getOrDefault(m.persistant(), "SCHED:PRIORITY", 20));

    LOG(INFO) << "Accept session with priority " << priority;

    m_laneMgr->requestLanes(std::move(layout), [&resp, priority,
                                                cb = std::move(cb), req = std::move(req), ectx = std::move(ectx),
                                                this](auto &&lanes) mutable {
        std::vector<tf::Device *> devices;

        CHECK(!lanes.empty());
        // add CPU device
        devices.emplace_back(m_laneMgr->compatibleCPUDevice());

        // prepare devices from lane
        for (auto &lane : lanes) {
            devices.emplace_back(lane->as_tfdevice());
        }

        // NOTE: laneId on ectx is separated from actual lane implementation.
        // It is only used to have separate scheduling domain. So use first lane's id as the id
        // Revisit if later multi-lane for a job is implemented.
        // TODO: support multiple lane id
        ectx->setLaneId(lanes.at(0)->id());

        auto session =
            std::make_shared<TFSession>(*this, ectx, std::move(devices), req->config(), req->mutable_graph_def());
        auto handle = session->handle();

        auto &lane = lanes.at(0);
        LOG(INFO) << "event: lane_assigned "
                       << as_json({
                              {"sess", handle},
                              {"laneId", lane->id()},
                              {"laneSize", lane->totalMemory()},
                              {"laneAvail", lane->availableMemory()},
                              {"laneStream", lane->baseStreamIndex()},
                          });
        // Keep a reference for lanes on ectx's user data
        // which should outlive the TFSession.
        ectx->setUserData(TFExecutionCtxData{std::forward<decltype(lanes)>(lanes), priority});

        // Register force interrupt handler
        ectx->setInterruptCallback([this, handle]() { popSession(handle)->safeClose(); });

        // Insert into the session map, which takes ownership of the session.
        {
            auto l = sstl::with_guard(m_mu);
            if (!m_sessions.try_emplace(handle, std::move(session)).second) {
                throw TFException(tf::errors::Internal("Error when inserting session ", handle));
            }
        }
        LOG(INFO) << "Accepting and created session " << handle;

        // reply
        resp.set_session_handle(handle);
        cb(Status::OK());
    });
}

std::shared_ptr<TFSession> TFInstance::findSession(const std::string &sessHandle)
{
    auto g = sstl::with_guard(m_mu);
    auto it = m_sessions.find(sessHandle);
    if (it == m_sessions.end()) {
        LOG(ERROR) << "Dumping all known sessions: ";
        for (auto &[h, ptr] : m_sessions) {
            LOG(ERROR) << " Session " << h << "@" << as_hex(ptr);
        }
        throw TFException(
            tf::errors::InvalidArgument("Session ", sessHandle, " is not found. Possibly, this master has restarted."));
    }
    return it->second;
}

std::shared_ptr<TFSession> TFInstance::popSession(const std::string &sessHandle)
{
    auto g = sstl::with_guard(m_mu);
    auto nh = m_sessions.extract(sessHandle);
    if (!nh) {
        throw TFException(
            tf::errors::InvalidArgument("Session ", sessHandle, " is not found. Possibly, this master has restarted."));
    }
    return std::move(nh.mapped());
}

void TFInstance::handleCloseSession(std::unique_ptr<tf::CloseSessionRequest> &&req, tf::CloseSessionResponse &resp,
                                    HandlerCallback &&cb)
{
    UNUSED(resp);
    popSession(req->session_handle())->deferClose(std::move(cb));
}

void TFInstance::handleListDevices(std::unique_ptr<tf::ListDevicesRequest> &&req, tf::ListDevicesResponse &resp,
                                   HandlerCallback &&cb)
{
    UNUSED(req);
    // This is never called
    UNUSED(resp);
    /*
    for (auto dev : devices()) {
        *(resp.add_local_device()) = dev->attributes();
    }
    */
    cb(Status::OK());
}

void TFInstance::handleReset(std::unique_ptr<tf::ResetRequest> &&req, tf::ResetResponse &resp, HandlerCallback &&cb)
{
    UNUSED(req);
    UNUSED(resp);
    std::vector<std::shared_ptr<TFSession>> sessToClose;
    {
        auto g = sstl::with_guard(m_mu);
        sessToClose.reserve(m_sessions.size());
        for (auto &[sessHandle, sess] : m_sessions) {
            UNUSED(sessHandle);
            sessToClose.emplace_back(std::move(sess));
        }
        m_sessions.clear();
    }

    Status s;
    for (auto &sess : sessToClose) {
        try {
            sess->safeClose();
        } catch (const TFException &ex) {
            s.Update(ex.code());
        }
    }
    sessToClose.clear();
    cb(s);
}

std::string TFInstance::maybeDumpGPUMemoryMap(tf::Device *dev) const
{
    if (dev->parsed_name().has_type && dev->parsed_name().type == tf::DEVICE_GPU) {
        auto alloc = dev->GetAllocator({});
        auto s = alloc->Name();
        if (alloc->Name().find("dbfc") != std::string::npos) {
            if (auto falloc = dynamic_cast<ForwardingAllocator*>(alloc)) {
                if (auto dbfc = dynamic_cast<tf::GPUDoubleBFCAllocator*>(falloc->base().get())) {
                    return dbfc->GenerateMemoryMap();
                }
            }
        }
    }
    return {};
}
} // namespace salus::oplib::tensorflow
