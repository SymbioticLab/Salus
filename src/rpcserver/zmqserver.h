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

#ifndef ZMQSERVER_H
#define ZMQSERVER_H

#include "irpcserver.h"

#include "zmq.hpp"

#include <vector>
#include <memory>
#include <thread>

class ServerWorker;
/**
 * @todo write docs
 */
class ZmqServer final : public IRpcServer
{
public:
    ZmqServer();

    ~ZmqServer() override;

    void start(std::unique_ptr<RpcServerCore> &&logic, const std::string &address, bool block = true) override;
    void join() override;
    void stop() override;

    size_t numWorkers() const;
    void setNumWorkers(size_t num);

private:
    void adjustNumWorkers(size_t num);
    void proxyLoop();

private:
    std::unique_ptr<RpcServerCore> m_pLogic;
    std::unique_ptr<std::thread> m_proxyLoopThread;

    zmq::context_t m_zmqCtx;
    zmq::socket_t m_frontend_sock;
    zmq::socket_t m_backend_sock;

    std::vector<std::unique_ptr<ServerWorker>> m_workers;
    size_t m_numWorkers;
    bool m_started;
};

#endif // ZMQSERVER_H
