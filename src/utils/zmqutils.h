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

#ifndef SALUS_SSTL_ZMQUTILS_H
#define SALUS_SSTL_ZMQUTILS_H

#include <vector>
#include <zmq.hpp>

namespace sstl {

class MultiPartMessage
{
public:
    MultiPartMessage();
    MultiPartMessage(MultiPartMessage &&other);
    explicit MultiPartMessage(std::vector<zmq::message_t> *ptr);
    MultiPartMessage(const MultiPartMessage &) = delete;

    MultiPartMessage &operator=(MultiPartMessage &&other);
    MultiPartMessage &operator=(const MultiPartMessage &) = delete;

    MultiPartMessage &merge(MultiPartMessage &&other);
    MultiPartMessage clone();

    size_t totalSize() const;

    std::vector<zmq::message_t> *release();

    std::vector<zmq::message_t> *operator->();

private:
    std::vector<zmq::message_t> m_parts;
};

} // namespace sstl

#endif // SALUS_SSTL_ZMQUTILS_H
