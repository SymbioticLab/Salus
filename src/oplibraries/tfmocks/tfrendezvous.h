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

class TFSession;

/**
 * @todo write docs
 */
class TFRendezvous : public tensorflow::Rendezvous
{
public:
    TFRendezvous(TFSession *sess);
    ~TFRendezvous() override;

    tensorflow::Status Send(const ParsedKey& parsed,
                            const Args& send_args,
                            const tensorflow::Tensor& val,
                            const bool is_dead) override;

    void RecvAsync(const ParsedKey& parsed, const Args& recv_args, DoneCallback done) override;

    void StartAbort(const tensorflow::Status& status) override;

private:
    TFSession *m_sess;
};

#endif // TFRENDEZVOUS_H
