//
// Created by peifeng on 2/12/18.
//

#include "dummysessionmgr.h"

namespace salus::oplib::tensorflow {

SingleSessionMgr::SingleSessionMgr(std::unique_ptr<tf::WorkerSession> &&workerSess)
    : m_workerSess(std::move(workerSess))
{
}

tf::WorkerSession *SingleSessionMgr::WorkerSessionForSession(const std::string &)
{
    return m_workerSess.get();
}

Status SingleSessionMgr::DeleteSession(const std::string &)
{
    return Status::OK();
}

Status SingleSessionMgr::CreateSession(const std::string &, const tf::ServerDef &, bool)
{
    return Status::OK();
}
} // namespace salus::oplib::tensorflow
