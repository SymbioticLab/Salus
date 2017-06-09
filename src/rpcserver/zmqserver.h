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

#include <boost/lockfree/queue.hpp>
#define BOOST_THREAD_VERSION 4
#include <boost/thread/future.hpp>

#include <vector>
#include <memory>
#include <thread>
#include <list>

class RpcServerCore;
class ServerWorker;

/**
 * @todo write docs
 */
class ZmqServer
{
public:
    typedef std::vector<zmq::message_t> MultiMessage;

    explicit ZmqServer(std::unique_ptr<RpcServerCore> &&logic);

    ~ZmqServer();

    /**
     * Start the server, must be called in the same thread as the constructor. Will blocks until
     * stop is called in another thread or ctrl-c signal received.
     */
    void start(const std::string &address);

    void requestStop();

    void join();

    /**
     * Low level api for sending messages back to client.
     */
    void sendMessage(std::unique_ptr<MultiMessage> &&parts);

private:
    void proxyLoop();
    void recvLoop();
    void sendLoop();

private:
    std::unique_ptr<std::thread> m_recvThread;
    std::unique_ptr<std::thread> m_sendThread;

    // Shared by proxy, recv, send threads
    constexpr static const char *m_baddr = "inproc://backend";
    zmq::context_t m_zmqCtx;
    bool m_keepRunning;

    // For the proxy loop
    zmq::socket_t m_frontend_sock;
    zmq::socket_t m_backend_sock;

    // For send loop
    struct SendItem {
        // we cannot use unique_ptr because boost::lockfree::queue does not support move semantics
        MultiMessage *p_parts;
    };
    boost::lockfree::queue<SendItem> m_sendQueue;

    // For recv loop
    std::unique_ptr<RpcServerCore> m_pLogic;
    std::list<boost::future<void>> m_futures;
};

#endif // ZMQSERVER_H
