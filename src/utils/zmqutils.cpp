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

#include "zmqutils.h"

namespace sstl {

MultiPartMessage::MultiPartMessage() {}

MultiPartMessage::MultiPartMessage(MultiPartMessage &&other)
    : m_parts(std::move(other.m_parts))
{
}
MultiPartMessage::MultiPartMessage(std::vector<zmq::message_t> *ptr)
    : m_parts(std::move(*ptr))
{
}

MultiPartMessage &MultiPartMessage::operator=(MultiPartMessage &&other)
{
    m_parts = std::move(other.m_parts);
    return *this;
}

MultiPartMessage &MultiPartMessage::merge(MultiPartMessage &&other)
{
    if (m_parts.empty()) {
        m_parts = std::move(other.m_parts);
    } else {
        m_parts.reserve(m_parts.size() + other.m_parts.size());
        std::move(std::begin(other.m_parts), std::end(other.m_parts), std::back_inserter(m_parts));
        other.m_parts.clear();
    }
    return *this;
}

MultiPartMessage MultiPartMessage::clone()
{
    MultiPartMessage mpm;
    for (auto &m : m_parts) {
        mpm->emplace_back();
        mpm->back().copy(&m);
    }
    return mpm;
}
size_t MultiPartMessage::totalSize() const
{
    size_t totalSize = 0;
    for (const auto &m : m_parts) {
        totalSize += m.size();
    }
    return totalSize;
}

std::vector<zmq::message_t> *MultiPartMessage::release()
{
    auto ptr = new std::vector<zmq::message_t>(std::move(m_parts));
    return ptr;
}

std::vector<zmq::message_t> *MultiPartMessage::operator->()
{
    return &m_parts;
}

} // namespace sstl
