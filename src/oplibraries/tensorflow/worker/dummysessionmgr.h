//
// Created by peifeng on 2/12/18.
//

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
