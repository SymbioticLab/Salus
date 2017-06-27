/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017  Aetf <aetf@unlimitedcodeworks.xyz>
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
 * 
 */

#ifndef TFOPLIBRARY_H
#define TFOPLIBRARY_H

#include "ioplibrary.h"

#include "executor.pb.h"
#include "tfoplibrary.pb.h"

#include <tensorflow/core/framework/op_segment.h>

#include <memory>
#include <atomic>
#include <mutex>

namespace tensorflow {
class OpKernel;
class DeviceMgr;
}

class TFSession;
class TFExecutionState;
class TFContext;

/**
 * @todo write docs
 */
class TFOpLibrary : public IOpLibrary
{
public:
    ~TFOpLibrary() override;

    bool accepts(const executor::OpKernelDef &operation) override;
    PTask createRunTask(ZmqServer::Sender sender,
                        const executor::EvenlopDef &evenlop,
                        const executor::RunRequest &request) override;

    PTask createRunGraphTask(ZmqServer::Sender sender,
                             const executor::EvenlopDef &evenlop,
                             const executor::RunGraphRequest &request) override;

    PTask createCustomTask(ZmqServer::Sender sender,
                           const executor::EvenlopDef &evenlop,
                           const executor::CustomRequest &push) override;


private:
    // subtasks for custom tasks
    /**
     * Called when remote rendezvous recv available
     */
    PTask createRendezRecvTask(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                               const executor::CustomRequest &req);

    /**
     * Called when new session created
     */
    PTask createInitSessionTask(ZmqServer::Sender sender,
                                const executor::EvenlopDef &evenlop,
                                const executor::CustomRequest &req);
    /**
     * Called when session closed
     */
    PTask createCloseSessionTask(ZmqServer::Sender sender,
                                 const executor::EvenlopDef &evenlop,
                                 const executor::CustomRequest &req);

    tensorflow::DeviceMgr *deviceManager() const { return m_deviceMgr.get(); };

private:
    TFSession *getOrCreateSession(const std::string &sess_id, const tensorflow::ConfigProto &cfgProto);
    TFSession *findSession(const std::string &sess_id);
    void destorySession(const std::string &sess_id);

    std::mutex m_mu; // protects m_sessions
    std::unordered_map<std::string, std::unique_ptr<TFSession>> m_sessions;

    std::atomic_uint_fast64_t m_sessionSeq;

    std::unique_ptr<tensorflow::DeviceMgr> m_deviceMgr;
};

class TFRunTask : public ITask
{
public:
    ~TFRunTask() override;

    TFRunTask(TFExecutionState *execState, ZmqServer::Sender &&sender,
              std::unique_ptr<tensorflow::NodeDef> &&nodedef, bool async,
              std::unique_ptr<executor::TFOpContextDef> &&tfctxdef);

    ProtoPtr run() override;

    bool isAsync() override;

    void runAsync(DoneCallback cb) override;

    bool prepare(DeviceType &dev) override;

private:
    TFExecutionState *m_exec;

    ZmqServer::Sender m_sender;

    bool m_async;

    std::unique_ptr<tensorflow::NodeDef> m_ndef;
    std::unique_ptr<executor::TFOpContextDef> m_tfctxdef;

    tensorflow::OpKernel *m_opkernel;
    std::shared_ptr<TFContext> m_context;
};

#endif // TFOPLIBRARY_H
