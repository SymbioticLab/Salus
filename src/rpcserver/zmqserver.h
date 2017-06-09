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

#include "zmq.hpp"

#include <vector>
#include <memory>
#include <thread>

class RpcServerCore;
class ServerWorker;

/**
 * @todo write docs
 */
class ZmqServer
{
public:
    explicit ZmqServer(std::unique_ptr<RpcServerCore> &&logic);

    ~ZmqServer();

    /**
     * Start the server, must be called in the same thread as the constructor. Will blocks until
     * stop is called in another thread or ctrl-c signal received.
     */
    void start(const std::string &address);

    void stop();

private:
    void proxyLoop();
    void recvLoop();
    void sendLoop();

private:
    std::unique_ptr<RpcServerCore> m_pLogic;

    std::unique_ptr<std::thread> m_recvThread;
    std::unique_ptr<std::thread> m_sendThread;

    zmq::context_t m_zmqCtx;
    zmq::socket_t m_frontend_sock;
    zmq::socket_t m_backend_sock;

    bool m_keepRunning;

    constexpr static const char *m_baddr = "inproc://backend";
};

#endif // ZMQSERVER_H
