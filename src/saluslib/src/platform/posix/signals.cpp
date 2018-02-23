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
 */

#include "platform/signals.h"

#include "platform/logging.h"

#include <signal.h>

#include <cstring>
#include <unordered_map>

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
void handler(int signo)
{
    const char *action = "";
    switch (signo) {
        case SIGINT:
        case SIGTERM:
            action = ", exiting";
            break;
        default:
            action = ", ignoring";
    }

    // Flush and start a new line on stdout, so ^C won't mess up the output
    std::cout << std::endl;
    LOG(INFO) << "Received signal " << signalName(signo) << action;
}

} // namespace

void initialize()
{
    installSignalHandler(SIGINT, handler);
    installSignalHandler(SIGTERM, handler);
}

void waitForTerminate()
{
    sigset_t set;
    sigemptyset(&set);
    sigsuspend(&set);
}

} // namespace signals
