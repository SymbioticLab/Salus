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

#ifndef SALUS_OPLIB_TENSORFLOW_DUMMYSESSIONMGR_H
#define SALUS_OPLIB_TENSORFLOW_DUMMYSESSIONMGR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include <functional>

namespace salus::oplib::tensorflow {

using CreateWorkerSessionFn = std::function<std::unique_ptr<tf::WorkerSession>(const std::string&)>;
class LocalSessionMgr : public tf::SessionMgrInterface
{
public:
    explicit LocalSessionMgr(CreateWorkerSessionFn fn);

    Status CreateSession(const std::string &session, const tf::ServerDef &server_def, bool isolate_session_state) override;

    tf::WorkerSession *WorkerSessionForSession(const std::string &session) override;

    Status DeleteSession(const std::string &session) override;

private:
    std::unique_ptr<tf::WorkerSession> m_workerSess;
    CreateWorkerSessionFn  m_fn;
};

}

#endif // SALUS_OPLIB_TENSORFLOW_DUMMYSESSIONMGR_H
