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
#include "protoutils/protoutils.h"

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

void ZmqServer::start(std::unique_ptr<RpcServerCore> logic, const std::string& address, bool block)
{
    m_pLogic = std::move(logic);

    try {
        INFO("Binding frontend socket to address: {}", address);
        m_frontend_sock.bind(address);

        auto baddr = "inproc://backend";
        DEBUG("Binding backend socket to address: {}", baddr);
        m_backend_sock.bind(baddr);
    } catch (zmq::error_t &err) {
        FATAL("Error while binding sockets: {}", err.what());
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
        ERROR("Exiting serving loop due to error: {}", err.what());
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
    INFO("ServerWorker started (thread {}), connecting to address: {}", m_thread->get_id(), baddr);
    m_sock.connect(baddr);

    try {
        while (!m_shouldStop) {
            zmq::message_t evenlop;
            zmq::message_t body;
            try {
                m_sock.recv(&evenlop);
                m_sock.recv(&body);
            } catch (zmq::error_t &err) {
                ERROR("Skipped one iteration due to error when receiving: {}", err.what());
                continue;
            }

            std::string type(static_cast<char *>(evenlop.data()), evenlop.size());
            auto pRequest = protoutils::createMessage(type, body.data(), body.size());
            if (!pRequest) {
                ERROR("Skipped one iteration due to malformatted request received. Evenlop data '{}'."
                      " Body size {}", type, body.size());
                continue;
            }
            auto pResponse = m_logic.dispatch(type, pRequest.get());

            zmq::message_t reply(pResponse->ByteSizeLong());
            pResponse->SerializeToArray(reply.data(), reply.size());

            try {
                m_sock.send(reply);
            } catch (zmq::error_t &err) {
                ERROR("Sending error when serving request {}: {}", type, err.what());
                continue;
            }
        }
    } catch (std::exception &e) {
        ERROR("Catched exception: {}", e.what());
    }
}
