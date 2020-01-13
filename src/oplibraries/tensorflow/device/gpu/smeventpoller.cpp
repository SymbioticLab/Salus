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

#include "smeventpoller.h"

#include "oplibraries/tensorflow/v3/smblocker.h"
#include "platform/thread_annotations.h"

namespace salus::oplib::tensorflow {

namespace {

} // namespace

SMEventPoller::SMEventPoller(tf::gpu::StreamExecutor *se)
    : m_pool(ThreadPoolOptions{}
             .setWorkerName("SMEvtWorker")
             // one thread for poller, one thread for executing callbacks
             .setNumThreads(2))
    , m_se(se)
{
    startPollingLoop();
}

SMEventPoller::~SMEventPoller()
{
    stopPollingLoop();

    // free anything owned by this
    for (auto &act : m_pendingActions) {
        SMBlocker::instance().release(act.count);
        if (act.func) {
            act.func();
        }
    }
}

void SMEventPoller::startPollingLoop()
{
    m_pool.run([this]() {
        pollLoop();
    });
}

void SMEventPoller::stopPollingLoop()
{
    m_stopPolling.notify();
    // make sure to wake up polling loop thread
    m_eventsStaging.notify();
    m_pollingStopped.wait();
}

void SMEventPoller::pollLoop()
{
    threading::set_thread_name("SMEvtPoller");
    // actions go from m_stagedEvents to staging, to waiting and finally to ready
    while (!m_stopPolling.notified()) {
        PendingActions staging;
        {
            auto g = sstl::with_guard(m_mu);
            staging.swap(m_stagedEvents);
        }

        m_pendingActions.insert(m_pendingActions.end(),
                                std::make_move_iterator(staging.begin()),
                                std::make_move_iterator(staging.end()));

        if (m_pendingActions.empty()) {
            m_eventsStaging.wait();
            continue;
        }

        auto ready = pollEvents();
        executeReady(ready);
    }
    m_pollingStopped.notify();
}

SMEventPoller::PendingActions SMEventPoller::pollEvents()
{
    if (VLOG_IS_ON(2)) {
        size_t freeSize;
        {
            auto g = sstl::with_guard(m_mu);
            freeSize = m_freeEvents.size();
        }
        VLOG(2) << "SMEventPoller m_freeEvents " << freeSize << " m_pendingActions " << m_pendingActions.size();
    }
    PendingActions ready;
    auto it = m_pendingActions.begin();
    while (it != m_pendingActions.end()) {
        auto &act = *it;
        CHECK_NOTNULL(act.event);
        auto s = act.event->PollForStatus();
        switch (s) {
        default:
        case tf::gpu::Event::Status::kUnknown:
        case tf::gpu::Event::Status::kError:
            // We don't expect to see these.  Someday maybe propagate
            // a Status error, but for now fail hard.
            LOG(FATAL) << "Unexpected Event status: " << static_cast<int>(s);
            break;
        case tf::gpu::Event::Status::kPending:
            break;
        case tf::gpu::Event::Status::kComplete:
            // add event back to free event
            {
                auto g = sstl::with_guard(m_mu);
                m_freeEvents.emplace_back(std::move(act.event));
            }
            // add action to ready
            ready.emplace_back(std::move(act));
            // remove from pending
            it = m_pendingActions.erase(it);
            // skip ++it
            continue;
        }

        ++it;
    }
    return ready;
}

void SMEventPoller::executeReady(SMEventPoller::PendingActions &ready)
{
    for (auto &act : ready) {
        SMBlocker::instance().release(act.count);
        if (act.func) {
            act.func();
        }
    }
}

void SMEventPoller::queueAction(tf::gpu::Stream *stream, PendingAction act)
{
    act.event = allocEvent();
    CHECK_NOTNULL(act.event);
    stream->ThenRecordEvent(act.event.get());

    {
        auto g = sstl::with_guard(m_mu);
        m_stagedEvents.emplace_back(std::move(act));
    }
    // Wake up the polling thread
    m_eventsStaging.notify();
}

std::unique_ptr<tf::gpu::Event> SMEventPoller::allocEvent()
{
    auto g = sstl::with_guard(m_mu);
    // Events are created on demand, and repeatedly reused.  There is no
    // limit placed here on the number of allocated Events.
    if (m_freeEvents.empty()) {
        m_freeEvents.emplace_back(std::make_unique<tf::gpu::Event>(m_se));
        m_freeEvents.back()->Init();
    }
    auto e = std::move(m_freeEvents.back());
    m_freeEvents.pop_back();
    return e;
}

} // namespace salus::oplib::tensorflow
