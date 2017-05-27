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
#include "zmqserver_p.h"

#include "rpcservercore.h"
#include "platform/logging.h"
#include "utils/protoutils.h"

#include "executor.pb.h"

#include <functional>
#include <chrono>
#include <iostream>

using namespace std::literals::chrono_literals;

ZmqServer::ZmqServer()
    : m_pLogic(nullptr)
    , m_zmqCtx(1)
    , m_frontend_sock(m_zmqCtx, ZMQ_ROUTER)
    , m_backend_sock(m_zmqCtx, ZMQ_DEALER)
    , m_numWorkers(std::thread::hardware_concurrency())
    , m_started(false)
{
    // Make sure we have at least 1 worker
    if (m_numWorkers <= 0) {
        m_numWorkers = 1;
    }
}

ZmqServer::~ZmqServer()
{
    stop();
}

size_t ZmqServer::numWorkers() const
{
    return m_workers.size();
}

void ZmqServer::setNumWorkers(size_t num)
{
    if (num <= 0) {
        num = 1;
    }
    m_numWorkers = num;

    if (m_started) {
        adjustNumWorkers(m_numWorkers);
    }
}

void ZmqServer::adjustNumWorkers(size_t num)
{
    // Only add new workers when started, otherwise m_plogic may be empty
    while (m_started && m_workers.size() < num) {
        std::unique_ptr<ServerWorker> worker(new ServerWorker(m_zmqCtx, *m_pLogic));
        worker->start();
        m_workers.push_back(std::move(worker));
    }
    while (m_workers.size() > num) {
        m_workers.back()->stop();
        m_workers.pop_back();
    }
}

void ZmqServer::start(std::unique_ptr<RpcServerCore> &&logic, const std::string& address, bool block)
{
    m_pLogic = std::move(logic);

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

    m_started = true;
    adjustNumWorkers(m_numWorkers);

    if (block) {
        join();
    }
}

void ZmqServer::join()
{
    INFO("Started serving loop");
    try {
        zmq::proxy(m_frontend_sock, m_backend_sock, nullptr);
    } catch (zmq::error_t &err) {
        ERR("Exiting serving loop due to error: {}", err);
    }
    stop();
}

void ZmqServer::stop()
{
    INFO("Stopping ZMQ context");
    m_started = false;
    m_zmqCtx.close();

    INFO("Stopping all workers");
    adjustNumWorkers(0);
    m_pLogic.reset();
}

ServerWorker::ServerWorker(zmq::context_t &ctx, RpcServerCore &logic)
    : m_thread(nullptr)
    , m_zmqCtx(ctx)
    , m_sock(m_zmqCtx, ZMQ_DEALER)
    , m_shouldStop(false)
    , m_logic(logic)
{}

void ServerWorker::start()
{
    m_thread.reset(new std::thread(std::bind(&ServerWorker::work, this)));
}

void ServerWorker::stop()
{
    m_shouldStop = true;
    m_thread->join();
}

void ServerWorker::work() {
    auto baddr = "inproc://backend";
    m_sock.connect(baddr);
    INFO("ServerWorker started (thread {}), connected to address: {}", m_thread->get_id(), baddr);

    try {
        while (!m_shouldStop) {
            std::vector<zmq::message_t> identities;
            zmq::message_t evenlop;
            zmq::message_t body;
            try {
                // First receive all identity frames added by ZMQ_ROUTER socket
                TRACE("Receiving identity frame {}", identities.size());
                identities.emplace_back();
                m_sock.recv(&identities.back());
                TRACE("Identity frame {}: {}", identities.size() - 1, identities.back());
                // Identity frames stop at an empty message
                // ZMQ_RCVMORE is a int64_t according to doc, not a bool
                while (identities.back().size() != 0 && m_sock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                    TRACE("Receiving identity frame {}", identities.size());
                    identities.emplace_back();
                    m_sock.recv(&identities.back());
                    TRACE("Identity frame {}: {}", identities.size() - 1, identities.back());
                }
                if (!m_sock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                    ERR("Skipped one iteration due to no body message part found after identity frames");
                    continue;
                }
                TRACE("Receiving body frames...");
                // Now receive our message
                m_sock.recv(&evenlop);
                m_sock.recv(&body);
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

            auto pResponse = m_logic.dispatch(type, pRequest.get());

            zmq::message_t reply(pResponse->ByteSizeLong());
            pResponse->SerializeToArray(reply.data(), reply.size());
            TRACE("Response proto object have size {}", reply.size());

            try {
                // First send out all saved identity frames, including the empty part
                for (auto &id : identities) {
                    m_sock.send(id, ZMQ_SNDMORE);
                }

                // Then send our message
                m_sock.send(reply);
                TRACE("Response sent");
            } catch (zmq::error_t &err) {
                ERR("Sending error when serving request {}: {}", type, err);
                continue;
            }
        }
    } catch (std::exception &e) {
        ERR("Catched exception: {}", e.what());
    }
}
