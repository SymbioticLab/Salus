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

#include "zmqserver.h"

#include "rpcservercore.h"
#include "platform/logging.h"
#include "utils/protoutils.h"

#include "executor.pb.h"

#include <functional>
#include <chrono>
#include <iostream>

using namespace std::literals::chrono_literals;

ZmqServer::ZmqServer(std::unique_ptr<RpcServerCore> &&logic)
    : m_pLogic(std::move(logic))
    , m_zmqCtx(1)
    , m_frontend_sock(m_zmqCtx, ZMQ_ROUTER)
    , m_backend_sock(m_zmqCtx, ZMQ_DEALER)
    , m_keepRunning(false)
{
}

ZmqServer::~ZmqServer()
{
    stop();
}

void ZmqServer::start(const std::string& address)
{
    if (m_keepRunning) {
        ERR("ZmqServer already started.");
        return;
    }

    try {
        INFO("Binding frontend socket to address: {}", address);
        m_frontend_sock.bind(address);

        auto baddr = "inproc://backend";
        DEBUG("Binding backend socket to address: {}", baddr);
        m_backend_sock.bind(baddr);
    } catch (zmq::error_t &err) {
        FATAL("Error while binding sockets: {}", err);
        // re-throw to stop the process
        throw;
    }

    m_keepRunning = true;
    m_recvThread = std::make_unique<std::thread>(std::bind(&ZmqServer::recvLoop, this));
    m_sendThread = std::make_unique<std::thread>(std::bind(&ZmqServer::sendLoop, this));

    // proxy loop must be called in the same thread as constructor (because of fe and bd sockets)
    proxyLoop();
}

void ZmqServer::proxyLoop()
{
    INFO("Started serving loop");
    try {
        zmq::proxy(m_frontend_sock, m_backend_sock, nullptr);
    } catch (zmq::error_t &err) {
        ERR("Exiting serving loop due to error: {}", err);
    }
}

void ZmqServer::recvLoop()
{
    zmq::socket_t sock(m_zmqCtx, ZMQ_DEALER);
    sock.connect(m_baddr);
    INFO("Recving loop started on thread {}", std::this_thread::get_id());

    while (m_keepRunning) {
        try {
            std::vector<zmq::message_t> identities;
            zmq::message_t evenlop;
            zmq::message_t body;
            try {
                TRACE("==============================================================");
                // First receive all identity frames added by ZMQ_ROUTER socket
                TRACE("Receiving identity frame {}", identities.size());
                identities.emplace_back();
                sock.recv(&identities.back());
                TRACE("Identity frame {}: {}", identities.size() - 1, identities.back());
                // Identity frames stop at an empty message
                // ZMQ_RCVMORE is a int64_t according to doc, not a bool
                while (identities.back().size() != 0 && sock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                    TRACE("Receiving identity frame {}", identities.size());
                    identities.emplace_back();
                    sock.recv(&identities.back());
                    TRACE("Identity frame {}: {}", identities.size() - 1, identities.back());
                }
                if (!sock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                    ERR("Skipped one iteration due to no body message part found after identity frames");
                    continue;
                }
                TRACE("Receiving body frames...");
                // Now receive our message
                sock.recv(&evenlop);
                sock.recv(&body);
            } catch (zmq::error_t &err) {
                ERR("Skipped one iteration due to error while receiving: {}", err);
                continue;
            }

            std::string type(static_cast<char *>(evenlop.data()), evenlop.size());
            DEBUG("Received request of proto type {}", type);
            DEBUG("Received request byte array size {}", body.size());

            auto pRequest = utils::createMessage(type, body.data(), body.size());
            if (!pRequest) {
                ERR("Skipped one iteration due to malformatted request received. Evenlop data '{}'."
                    " Body size {}", type, body.size());
                continue;
            }
            TRACE("Created request proto object from message at {:x}",
                  reinterpret_cast<uint64_t>(pRequest.get()));

            auto pResponse = m_pLogic->dispatch(type, pRequest.get());
            if (!pResponse) {
                ERR("Skipped to send one reply due to error in logic dispatch");
                continue;
            }

            zmq::message_t reply(pResponse->ByteSizeLong());
            pResponse->SerializeToArray(reply.data(), reply.size());
            TRACE("Response proto object have size {}", reply.size());

            try {
                // First send out all saved identity frames, including the empty part
                for (auto &id : identities) {
                    sock.send(id, ZMQ_SNDMORE);
                }

                // Then send our message
                sock.send(reply);
                TRACE("Response sent");
            } catch (zmq::error_t &err) {
                ERR("Sending error when serving request {}: {}", type, err);
                continue;
            }
        } catch (std::exception &e) {
            ERR("Caught exception in recv loop: {}", e.what());
        }
    }
}

void ZmqServer::sendLoop()
{
    zmq::socket_t sock(m_zmqCtx, ZMQ_DEALER);
    sock.connect(m_baddr);
    INFO("Sending loop started on thread {}", std::this_thread::get_id());

    while (m_keepRunning) {
        
    }
}

void ZmqServer::stop()
{
    if (!m_keepRunning) {
        return;
    }

    INFO("Stopping ZMQ context");
    m_keepRunning = false;
    m_zmqCtx.close();
}
