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
    : m_zmqCtx(1)
    , m_keepRunning(false)
    , m_frontend_sock(m_zmqCtx, ZMQ_ROUTER)
    , m_backend_sock(m_zmqCtx, ZMQ_PAIR)
    , m_sendQueue(128)
    , m_pLogic(std::move(logic))
{
}

ZmqServer::~ZmqServer()
{
    requestStop();
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
//     m_recvThread = std::make_unique<std::thread>(std::bind(&ZmqServer::recvLoop, this));
    m_sendThread = std::make_unique<std::thread>(std::bind(&ZmqServer::sendLoop, this));

    // proxy and recving loop must be called in the same thread as constructor (because of fe and bd sockets)
    halfProxyLoop();
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

int ZmqServer::pollWithCheck(zmq::pollitem_t *items, size_t nitem, long timeout)
{
    int rc = 0;
    try {
        rc = zmq::poll(items, nitem, timeout);
    } catch (zmq::error_t &err) {
        switch (err.num()) {
        case ETIMEDOUT:
            return 0;
        case EINTR:
        case ETERM:
            m_keepRunning = false;
            return -1;
        default:
            ERR("Exiting serving due to error while polling: {}", err);
            return -1;
        }
    }
    return rc;
}

void ZmqServer::halfProxyLoop()
{
    INFO("Started recving and sending loop");
    // set up pulling.
    // we are interested in POLLIN and POLLOUT on m_frontend_sock, and POLLIN out m_backend_sock.
    // messages received on m_frontend_sock are directly dispatched using m_pLogic,
    // messages received on m_backend_sock are forwarded to m_frontend_sock.

    zmq::pollitem_t pollin_events[] = {
        {&m_frontend_sock, 0, ZMQ_POLLIN, 0},
        {&m_backend_sock, 0, ZMQ_POLLIN, 0},
    };
    zmq::pollitem_t all_events[] = {
        {&m_frontend_sock, 0, ZMQ_POLLIN | ZMQ_POLLOUT, 0},
        {&m_backend_sock, 0, ZMQ_POLLIN, 0},
    };
    zmq::pollitem_t *wait_events = pollin_events;

    bool canSendOut = false;
    bool needSendOut = false;
    bool shouldDispatch = false;
    while (m_keepRunning) {
        // first blocking wait on pollin (read) events
        int rc = 0;
        rc = pollWithCheck(wait_events, 2, -1);
        if (rc < 0) {
            break;
        }
        // something happened, so we poll w/o waiting on all_events
        // to set events in all_events
        rc = pollWithCheck(all_events, 2, -1);
        if (rc < 0) {
            break;
        }

        // process events
        for (int i = 0; i != rc; i++) {
            auto revents = all_events[i].revents;
            if (all_events[i].socket == &m_frontend_sock) {
                shouldDispatch = (revents & ZMQ_POLLIN) != 0;
                canSendOut = (revents & ZMQ_POLLOUT) != 0;
            } else if (all_events[i].socket == &m_backend_sock) {
                needSendOut = (revents & ZMQ_POLLIN) != 0;
            }
        }

        // process dispatch if any
        if (shouldDispatch) {
            dispatch(m_frontend_sock);
            shouldDispatch = false;
        }

        // forward any send message
        if (needSendOut && canSendOut) {
            TRACE("Forwarding message out");
            try {
                zmq::message_t msg;
                while (true) {
                    m_backend_sock.recv(&msg);
                    TRACE("Forwarding message part: {}", msg);
                    bool more = m_backend_sock.getsockopt<int64_t>(ZMQ_RCVMORE);
                    m_frontend_sock.send(&msg, more ? ZMQ_SNDMORE : 0);
                    if (!more)
                        break;
                }
            } catch (zmq::error_t &err) {
                ERR("Dropping message while sending out due to error: {}", err);
            }
            needSendOut = canSendOut = false;
            wait_events = pollin_events;
        } else if (needSendOut) {
            // should also wait for POLLOUT on m_frontend_sock
            wait_events = all_events;
        } else if (canSendOut) {
            // only wait for POLLIN on sockets
            wait_events = pollin_events;
        }
    }
}

void ZmqServer::dispatch(zmq::socket_t &sock)
{
    auto identities = std::make_unique<MultiMessage>();
    zmq::message_t evenlop;
    zmq::message_t body;
    try {
        TRACE("==============================================================");
        // First receive all identity frames added by ZMQ_ROUTER socket
        TRACE("Receiving identity frame {}", identities->size());
        identities->emplace_back();
        sock.recv(&identities->back());
        TRACE("Identity frame {}: {}", identities->size() - 1, identities->back());
        // Identity frames stop at an empty message
        // ZMQ_RCVMORE is a int64_t according to doc, not a bool
        while (identities->back().size() != 0 && sock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
            TRACE("Receiving identity frame {}", identities->size());
            identities->emplace_back();
            sock.recv(&identities->back());
            TRACE("Identity frame {}: {}", identities->size() - 1, identities->back());
        }
        if (!sock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
            ERR("Skipped one iteration due to no body message part found after identity frames");
            return;
        }
        TRACE("Receiving body frames...");
        // Now receive our message
        sock.recv(&evenlop);
        sock.recv(&body);
    } catch (zmq::error_t &err) {
        ERR("Skipped one iteration due to error while receiving: {}", err);
        return;
    }
    std::string type(static_cast<char *>(evenlop.data()), evenlop.size());
    DEBUG("Received request of proto type {}", type);
    DEBUG("Received request byte array size {}", body.size());

    auto pRequest = utils::createMessage(type, body.data(), body.size());
    if (!pRequest) {
        ERR("Skipped one iteration due to malformatted request received. Evenlop data '{}'."
            " Body size {}", type, body.size());
        return;
    }
    TRACE("Created request proto object from message at {:x}",
            reinterpret_cast<uint64_t>(pRequest.get()));

    auto f = m_pLogic->dispatch(type, pRequest.get())

    .then(boost::launch::inherit, [parts = std::move(identities), this](auto f) mutable {
        INFO("callback called in thread {}", std::this_thread::get_id());
        ProtoPtr pResponse;
        try {
            pResponse = f.get();
        } catch(std::exception &ex) {
            ERR("Caught exception in logic dispatch: {}", ex.what());
        }
        if (!pResponse) {
            ERR("Skipped to send one reply due to error in logic dispatch");
            return;
        }

        parts->emplace_back(pResponse->ByteSizeLong());
        auto &reply = parts->back();
        pResponse->SerializeToArray(reply.data(), reply.size());
        TRACE("Response proto object have size {}", reply.size());
        this->sendMessage(std::move(parts));
    });

    // save the future so it won't deconstrcut, which will block.
    m_futures.push_back(std::move(f));
    // TODO: do the clean up in a seprate thread? And smarter clean up?
    cleanupFutures();

    return;
}

void ZmqServer::recvLoop()
{
    zmq::socket_t sock(m_zmqCtx, ZMQ_DEALER);
    sock.connect(m_baddr);
    INFO("Recving loop started on thread {}", std::this_thread::get_id());

    while (m_keepRunning) {
        try {
            auto identities = std::make_unique<MultiMessage>();
            zmq::message_t evenlop;
            zmq::message_t body;
            try {
                TRACE("==============================================================");
                // First receive all identity frames added by ZMQ_ROUTER socket
                TRACE("Receiving identity frame {}", identities->size());
                identities->emplace_back();
                sock.recv(&identities->back());
                TRACE("Identity frame {}: {}", identities->size() - 1, identities->back());
                // Identity frames stop at an empty message
                // ZMQ_RCVMORE is a int64_t according to doc, not a bool
                while (identities->back().size() != 0 && sock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                    TRACE("Receiving identity frame {}", identities->size());
                    identities->emplace_back();
                    sock.recv(&identities->back());
                    TRACE("Identity frame {}: {}", identities->size() - 1, identities->back());
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

            auto f = m_pLogic->dispatch(type, pRequest.get())

            .then(boost::launch::inherit, [parts = std::move(identities), this](auto f) mutable {
                INFO("callback called in thread {}", std::this_thread::get_id());
                ProtoPtr pResponse;
                try {
                    pResponse = f.get();
                } catch(std::exception &ex) {
                    ERR("Caught exception in logic dispatch: {}", ex.what());
                }
                if (!pResponse) {
                    ERR("Skipped to send one reply due to error in logic dispatch");
                    return;
                }

                parts->emplace_back(pResponse->ByteSizeLong());
                auto &reply = parts->back();
                pResponse->SerializeToArray(reply.data(), reply.size());
                TRACE("Response proto object have size {}", reply.size());
                this->sendMessage(std::move(parts));
            });

            // save the future so it won't deconstrcut, which will block.
            m_futures.push_back(std::move(f));
            // TODO: do the clean up in a seprate thread? And smarter clean up?
            cleanupFutures();
        } catch (std::exception &e) {
            ERR("Caught exception in recv loop: {}", e.what());
        }
    }
}

void ZmqServer::cleanupFutures()
{
    while (m_futures.front().is_ready()) {
        m_futures.pop_front();
    }
}

void ZmqServer::sendMessage(std::unique_ptr<MultiMessage> &&parts)
{
    m_sendQueue.push({parts.release()});
}

void ZmqServer::sendLoop()
{
    zmq::socket_t sock(m_zmqCtx, ZMQ_PAIR);
    sock.connect(m_baddr);
    INFO("Sending loop started on thread {}", std::this_thread::get_id());

    while (m_keepRunning) {
        SendItem item;
        if(!m_sendQueue.pop(item)) {
            std::this_thread::sleep_for(1ms);
            continue;
        }
        // Wrap the address in smart pointer immediately so we won't risk memory leak.
        std::unique_ptr<MultiMessage> parts(item.p_parts);
        try {
            for (size_t i = 0; i != parts->size() - 1; ++i) {
                auto &msg = parts->at(i);
                sock.send(msg, ZMQ_SNDMORE);
            }
            sock.send(parts->back());
            TRACE("Response sent on internal socket");
        } catch (zmq::error_t &err) {
            ERR("Sending error: {}", err);
        }
    }
}

void ZmqServer::requestStop()
{
    if (!m_keepRunning) {
        return;
    }

    INFO("Stopping ZMQ context");
    m_keepRunning = false;
    m_zmqCtx.close();
}

void ZmqServer::join()
{
    if (m_recvThread && m_recvThread->joinable()) {
        m_recvThread->join();
    }
    if (m_sendThread && m_sendThread->joinable()) {
        m_sendThread->join();
    }
}
