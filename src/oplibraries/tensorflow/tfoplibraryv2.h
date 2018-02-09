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

#ifndef TFOPLIBRARYV2_H
#define TFOPLIBRARYV2_H

#include "execution/executionengine.h"
#include "oplibraries/ioplibrary.h"
#include <unordered_map>

namespace executor {
class CustomRequest;
} // namespace executor

/**
 * @brief TFOpLibrary that uses TFInstance internally
 */
class TFOpLibraryV2 : public IOpLibrary
{
public:
    TFOpLibraryV2() = default;
    ~TFOpLibraryV2();

    bool initialize() override;
    void uninitialize() override;

    bool accepts(const executor::OpKernelDef &operation) override;

    void onCustom(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                  const executor::CustomRequest &req, DoneCallback cb) override;

    void onRunGraph(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop,
                    const executor::RunGraphRequest &req, DoneCallback cb) override;

    void onRun(ZmqServer::Sender sender, const executor::EvenlopDef &evenlop, const executor::RunRequest &req,
               DoneCallback cb) override;

private:
    const std::string &sessionFromRecvId(const std::string &recvId);

    // A map for last seen session per client process
    // recvId -> sessHandle
    std::unordered_map<std::string, std::string> m_lastSession;

    size_t m_maxOpenSessions;
};

#endif // TFOPLIBRARYV2_H
