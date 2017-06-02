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

namespace tensorflow {
class OpKernel;
}

class TFSession;
class TFContext;

/**
 * @todo write docs
 */
class TFOpLibrary : public IOpLibrary
{
public:
    ~TFOpLibrary() override;

    bool accepts(const executor::OpKernelDef &operation) override;
    std::unique_ptr<ITask> createRunTask(const executor::OpKernelDef &opdef,
                                         const executor::OpContextDef &ctxdef) override;

    std::unique_ptr<ITask> createFetchTask(const executor::FetchRequest &fetch) override;
    std::unique_ptr<ITask> createPushTask(const executor::PushRequest &push) override;

private:
    TFSession *getOrCreateSession(const std::string &sess_id, int graph_def_version,
                                  const tensorflow::ConfigProto &cfgProto,
                                  const tensorflow::FunctionDefLibrary &fDefLib);
    TFSession *getSession(const std::string &sess_id);

    std::mutex m_mu; // protects m_sessions
    std::unordered_map<std::string, std::unique_ptr<TFSession>> m_sessions;
};

class TFRunTask : public ITask
{
public:
    ~TFRunTask() override;

    TFRunTask(TFSession *session, std::unique_ptr<tensorflow::OpKernel> &&kernel,
              std::unique_ptr<TFContext> &&context);

    executor::Status run(google::protobuf::Message *out) override;

private:
    std::unique_ptr<tensorflow::OpKernel> m_opkernel;
    std::unique_ptr<TFContext> m_context;

    TFSession *m_session;
};

class TFFetchTask : public ITask
{
public:
    ~TFFetchTask() override;

    TFFetchTask(TFSession *session, std::unique_ptr<executor::TFTensors> &&tensors);

    executor::Status run(google::protobuf::Message *out) override;

private:
    std::unique_ptr<executor::TFTensors> m_tensors;

    TFSession *m_session;
};

class TFPushTask : public ITask
{
public:
    ~TFPushTask() override;

    TFPushTask(TFSession *session, std::unique_ptr<executor::TFPushRequest> &&tensors);

    executor::Status run(google::protobuf::Message *out) override;

private:
    std::unique_ptr<executor::TFPushRequest> m_tensors;

    TFSession *m_session;
};
#endif // TFOPLIBRARY_H
