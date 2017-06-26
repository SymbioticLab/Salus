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

#ifndef TFRENDEZVOUS_H
#define TFRENDEZVOUS_H

#include <tensorflow/core/framework/rendezvous.h>
#include <tensorflow/core/platform/mutex.h>

#include <unordered_map>

class TFExecutionState;

/**
 * TODO: TFRendezvous need not to be thread safe as it is only used per TFContext
 */
class TFRendezvous : public tensorflow::Rendezvous
{
public:
    struct SendItem {
        Args args;
        bool isDead;
        tensorflow::Tensor val;

        SendItem();
        SendItem(const Args &a, bool d, tensorflow::Tensor &&v);
    };
    using SentTensorTable = std::unordered_map<std::string, SendItem>;

    struct RecvItem {
        Args args;

        RecvItem();
        explicit RecvItem(const Args &a);
    };
    using RecvTable = std::unordered_map<std::string, RecvItem>;

    explicit TFRendezvous(TFExecutionState *exec);
    ~TFRendezvous() override;


    tensorflow::Status Send(const ParsedKey& parsed,
                            const Args& send_args,
                            const tensorflow::Tensor& val,
                            const bool is_dead) override;

    void RecvAsync(const ParsedKey& parsed, const Args& recv_args, DoneCallback done) override;

    void StartAbort(const tensorflow::Status& status) override;

    /**
     * Release pending sent tensors
     */
    SentTensorTable releasePendingSentTensors();

    /**
     * Release pending recv requests
     */
    RecvTable releasePendingRecv();

    /**
     * Send update got from RecvUpdate, this send is not recorded as pending sent tensors.
     */
    tensorflow::Status triggerSend(const ParsedKey& parsed,
                                   const Args& send_args,
                                   const tensorflow::Tensor& val,
                                   const bool is_dead);

private:
    TFExecutionState *m_exec;
    tensorflow::Rendezvous *m_local;

    mutable tensorflow::mutex m_mu;
    SentTensorTable m_tensors GUARDED_BY(m_mu);

    mutable tensorflow::mutex m_recvmu;
    RecvTable m_recv GUARDED_BY(m_recvmu);
};

#endif // TFRENDEZVOUS_H
