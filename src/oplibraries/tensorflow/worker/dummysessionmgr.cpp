//
// Created by peifeng on 2/12/18.
//

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
