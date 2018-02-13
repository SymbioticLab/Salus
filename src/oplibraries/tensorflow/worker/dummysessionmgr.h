//
// Created by peifeng on 2/12/18.
//

#ifndef SALUS_OPLIB_TENSORFLOW_DUMMYSESSIONMGR_H
#define SALUS_OPLIB_TENSORFLOW_DUMMYSESSIONMGR_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

namespace salus::oplib::tensorflow {

class SingleSessionMgr : public tf::SessionMgrInterface
{
public:
    explicit SingleSessionMgr(std::unique_ptr<tf::WorkerSession> &&workerSess);

    Status CreateSession(const std::string &session, const tf::ServerDef &server_def, bool isolate_session_state) override;

    tf::WorkerSession *WorkerSessionForSession(const std::string &session) override;

    Status DeleteSession(const std::string &session) override;

private:
    std::unique_ptr<tf::WorkerSession> m_workerSess;
};

}

#endif // SALUS_OPLIB_TENSORFLOW_DUMMYSESSIONMGR_H
