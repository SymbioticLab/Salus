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
 */

#ifndef TFOPLIBRARYV2_H
#define TFOPLIBRARYV2_H

#include "oplibraries/ioplibrary.h"

#include <atomic>
#include <memory>
#include <mutex>

namespace tensorflow {
namespace remote {

class TFOpLibraryProxy;
class TFSessionProxy;

} // namespace remote
} // namespace tensorflow

namespace executor {
class CustomRequest;
} // namespace executor

/**
 * TFOpLibrary that uses MasterSession internally
 */
class TFOpLibraryV2 : public IOpLibrary
{
public:
    /**
     * Default constructor
     */
    TFOpLibraryV2();

    /**
     * Destructor
     */
    ~TFOpLibraryV2() override;

    bool accepts(const executor::OpKernelDef &operation) override;

    PTask createCustomTask(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                           const executor::CustomRequest &req) override;

    PTask createRunGraphTask(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                             const executor::RunGraphRequest &req) override;

    PTask createRunTask(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                        const executor::RunRequest &req) override;

private:
    using Proxy = tensorflow::remote::TFSessionProxy;

    Proxy *getOrCreateProxy(const std::string &recvId);
    void deregisterProxy(const std::string &recvId);

    void handleCloseSession(const std::string &recvId, Proxy *proxy, const executor::CustomRequest&, ITask::DoneCallback);

    std::mutex m_mu;
    std::unordered_map<std::string, std::unique_ptr<Proxy>> m_proxies;

    std::unique_ptr<tensorflow::remote::TFOpLibraryProxy> m_proxy;

    size_t m_maxOpenSessions;
};

#endif // TFOPLIBRARYV2_H
