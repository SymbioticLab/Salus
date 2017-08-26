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

#ifndef ZMQUTILS_H
#define ZMQUTILS_H

#include <zmq.hpp>

#include <vector>

namespace utils {

class MultiPartMessage
{
public:
    MultiPartMessage();
    MultiPartMessage(MultiPartMessage &&other);
    MultiPartMessage(std::vector<zmq::message_t> *ptr);
    MultiPartMessage(const MultiPartMessage&) = delete;

    MultiPartMessage &operator=(MultiPartMessage &&other);
    MultiPartMessage &operator=(const MultiPartMessage&) = delete;

    MultiPartMessage &merge(MultiPartMessage &&other);
    MultiPartMessage clone();

    size_t totalSize() const;

    std::vector<zmq::message_t> *release();

    std::vector<zmq::message_t> *operator->();

private:
    std::vector<zmq::message_t> m_parts;
};

} // namespace utils

#endif // ZMQUTILS_H
