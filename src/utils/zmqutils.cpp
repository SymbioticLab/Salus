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
