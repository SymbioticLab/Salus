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

#ifndef ZMQSERVER_H
#define ZMQSERVER_H

#include "rpcserver/iothreadpool.h"
#include "utils/protoutils.h"
#include "utils/zmqutils.h"

#include <zmq.hpp>

#include <boost/lockfree/queue.hpp>

#include <atomic>
#include <vector>
#include <memory>
#include <thread>
#include <list>

using sstl::MultiPartMessage;

class RpcServerCore;

/**
 * @todo write docs
 */
class ZmqServer
{
public:
    ZmqServer();

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

        template<typename Func>
        auto post(Func &&f)
        {
            return m_server.m_iopool.post(std::forward<Func>(f));
        }

        template<typename Func>
        auto defer(Func &&f)
        {
            return m_server.m_iopool.defer(std::forward<Func>(f));
        }

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
    void proxyRecvLoop(const std::string &feAddr);

    /**
     * Poll on items with check
     */
    bool pollWithCheck(std::vector<zmq::pollitem_t> &items, long timeout);

    /**
     * Read a whole message from m_frontend_sock, and dispatch using m_pLogic
     */
    void dispatch(zmq::socket_t &sock);

private:
    // Pool to place blocking operations
    salus::IOThreadPool m_iopool;

    // Shared by proxy&recv and send threads
    zmq::context_t m_zmqCtx;
    std::atomic_bool m_keepRunning;

    // For the proxy&recv loop
    std::unique_ptr<std::thread> m_recvThread;

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
