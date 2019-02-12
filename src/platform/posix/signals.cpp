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

#include "platform/signals.h"

#include "platform/logging.h"

#include <atomic>
#include <csignal>
#include <cstring>
#include <cstdio>
#include <cerrno>

namespace signals {

void installSignalHandler(int sig, Handler h)
{
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    sigemptyset(&act.sa_mask);

    act.sa_handler = h;

    auto err = sigaction(sig, &act, nullptr);
    if (err) {
        LOG(ERROR) << "Error when setup signal handler: " << err;
    }
}

const char *signalName(int sig)
{
#define CASE(name)                                                                                           \
    case name:                                                                                               \
        do {                                                                                                 \
            return #name;                                                                                    \
        } while (false)

    switch (sig) {
        CASE(SIGHUP);
        CASE(SIGINT);
        CASE(SIGILL);
        CASE(SIGABRT);
        CASE(SIGFPE);
        CASE(SIGKILL);
        CASE(SIGPIPE);
        CASE(SIGALRM);
        CASE(SIGTERM);
        CASE(SIGUSR1);
        CASE(SIGUSR2);
        CASE(SIGCHLD);
        CASE(SIGCONT);
        CASE(SIGSTOP);
        CASE(SIGTSTP);
        CASE(SIGTTIN);
        CASE(SIGTTOU);
    default:
        return "Unknown";
    }

#undef CASE
}

namespace {

std::atomic_int gTheSignal;

std::atomic<SignalAction> gSignalAction;

extern "C" void handler(int signo)
{
    gTheSignal = signo;
    switch (signo) {
        case SIGINT:
        case SIGTERM:
            gSignalAction = SignalAction::Exit;
            break;
        default:
            gSignalAction = SignalAction::Ignore;
    }

    // Flush and start a new line on stdout, so ^C won't mess up the output
    auto esaved = errno;
    printf("\n");
    fflush(stdout);
    errno = esaved;
}

} // namespace

void initialize()
{
    gTheSignal = 0;
    gSignalAction = SignalAction::Ignore;

    installSignalHandler(SIGINT, handler);
    installSignalHandler(SIGTERM, handler);
}

std::pair<int, SignalAction> waitForTerminate()
{
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGTERM);
    sigaddset(&set, SIGINT);
    int sig;
    sigwait(&set, &sig);

    LOG(INFO) << "Received signal " << signalName(sig) << "(" << sig << ")";

    return {sig, SignalAction::Exit};
}

} // namespace signals
