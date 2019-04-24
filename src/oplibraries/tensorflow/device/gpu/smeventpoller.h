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

#ifndef SALUS_OPLIB_TENSORFLOW_SMEVENTPOLLER_H
#define SALUS_OPLIB_TENSORFLOW_SMEVENTPOLLER_H

#include "oplibraries/tensorflow/tensorflow_headers.h"

#include "execution/threadpool/threadpool.h"
#include "utils/fixed_function.hpp"
#include "utils/threadutils.h"
#include "utils/pointerutils.h"

#include <vector>
#include <memory>
#include <list>

namespace salus::oplib::tensorflow {

class SMEventPoller
{
public:
    explicit SMEventPoller(tf::gpu::StreamExecutor *se);
    ~SMEventPoller();

    inline void thenReleaseSM(tf::gpu::Stream *stream, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        queueAction(stream, {count, {}, nullptr});
    }

    inline void thenExecute(tf::gpu::Stream *stream, sstl::FixedFunction<void()> func)
    {
        queueAction(stream, {{}, std::move(func), nullptr});
    }

private:
    // Posting action from other threads
    struct PendingAction
    {
        uint64_t count; // num of SMs to release
        sstl::FixedFunction<void()> func; // action to execute
        std::unique_ptr<tf::gpu::Event> event; // perform action after this event
    };

    using PendingActions = std::vector<PendingAction>;

    std::unique_ptr<tf::gpu::Event> allocEvent();

    void queueAction(tf::gpu::Stream *stream, PendingAction action);

    void startPollingLoop();
    void stopPollingLoop();

    void pollLoop();
    PendingActions pollEvents();
    void executeReady(PendingActions &ready);

    // pending actions waiting for its events, in order
    std::list<PendingAction> m_pendingActions;

    // Threading related variables
    sstl::notification m_stopPolling;
    sstl::notification m_pollingStopped;

    ThreadPool m_pool;

    // other threads put actions into this queue, which will be regularly picked up by polling thread
    PendingActions m_stagedEvents GUARDED_BY(m_mu);
    std::mutex m_mu;
    sstl::notification m_eventsStaging;

    // GPU Event related variables
    tf::gpu::StreamExecutor * const m_se;

    // Free events
    std::vector<std::unique_ptr<tf::gpu::Event>> m_freeEvents GUARDED_BY(m_mu);
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_SMEVENTPOLLER_H
