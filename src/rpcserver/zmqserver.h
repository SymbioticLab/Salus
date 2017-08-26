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

#include "utils/protoutils.h"
#include "utils/zmqutils.h"

#include <zmq.hpp>

#include <boost/lockfree/queue.hpp>

#include <vector>
#include <memory>
#include <thread>
#include <list>

using utils::MultiPartMessage;

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

    void requestStop();

    void join();

    class SenderImpl
    {
    public:
        SenderImpl(ZmqServer &server, uint64_t seq, MultiPartMessage &&m_identities);

        void sendMessage(ProtoPtr &&msg);
        void sendMessage(const std::string &typeName, MultiPartMessage &&msg);

        uint64_t sequenceNumber() const;

    private:
        ZmqServer &m_server;
        MultiPartMessage m_identities;
        uint64_t m_seq;
    };
    using Sender = std::shared_ptr<SenderImpl>;

private:
    /**
     * Low level api for sending messages back to client.
     */
    void sendMessage(MultiPartMessage &&parts);

    void sendLoop();
    void proxyRecvLoop();

    /**
     * Poll on items with check
     */
    bool pollWithCheck(const std::vector<zmq::pollitem_t> &items, long timeout);

    /**
     * Read a whole message from m_frontend_sock, and dispatch using m_pLogic
     */
    void dispatch(zmq::socket_t &sock);

private:
    // Shared by proxy&recv and send threads
    constexpr static const char *m_baddr = "inproc://backend";
    zmq::context_t m_zmqCtx;
    bool m_keepRunning;

    // For the proxy&recv loop
    zmq::socket_t m_frontend_sock;
    zmq::socket_t m_backend_sock;

    std::unique_ptr<RpcServerCore> m_pLogic;

    // For send loop
    std::unique_ptr<std::thread> m_sendThread;
    struct SendItem {
        // we cannot use unique_ptr because boost::lockfree::queue does not support move semantics
        std::vector<zmq::message_t> *p_parts;
    };
    boost::lockfree::queue<SendItem> m_sendQueue;
};

#endif // ZMQSERVER_H
