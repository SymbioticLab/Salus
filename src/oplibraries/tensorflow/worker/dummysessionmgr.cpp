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

#include "dummysessionmgr.h"

namespace salus::oplib::tensorflow {

LocalSessionMgr::LocalSessionMgr(CreateWorkerSessionFn fn)
    : m_workerSess(nullptr)
    , m_fn(std::move(fn))
{
}

tf::WorkerSession *LocalSessionMgr::WorkerSessionForSession(const std::string &)
{
    DCHECK(m_workerSess);
    return m_workerSess.get();
}

Status LocalSessionMgr::DeleteSession(const std::string &)
{
    return Status::OK();
}

Status LocalSessionMgr::CreateSession(const std::string &session, const tf::ServerDef &, bool)
{
    if (!m_workerSess) {
        m_workerSess = m_fn(session);
    }

    return Status::OK();
}
} // namespace salus::oplib::tensorflow
