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

#ifndef IRPCSERVER_H
#define IRPCSERVER_H

#include <memory>
#include <string>

class RpcServerCore;

/**
 * @todo write docs
 */
class IRpcServer
{
public:
    virtual ~IRpcServer();

    virtual void start(std::unique_ptr<RpcServerCore> logic, const std::string &address, bool block = true) = 0;
    virtual void join() = 0;
    virtual void stop() = 0;
};

#endif // IRPCSERVER_H
