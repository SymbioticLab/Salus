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

#ifndef PROTOUTILS_H
#define PROTOUTILS_H

#include "utils/pointerutils.h"

#ifndef NDEBUG
#define NDEBUG
#define NEED_UNDEF_NDEBUG
#endif

#include <google/protobuf/message.h>

#ifdef NEED_UNDEF_NDEBUG
#undef NDEBUG
#undef NEED_UNDEF_NDEBUG
#endif

#include <memory>

using ProtoPtr = std::unique_ptr<::google::protobuf::Message>;

namespace symbiotic::salus {
/**
 * Create the protobuf message of specific type name `type` from a byte buffer `data` of length `len`.
 *
 * @return created Message, or nullptr if specified type not found or data is malformatted.
 */
ProtoPtr createMessage(const std::string &type, const void *data, size_t len);

template<typename T>
std::unique_ptr<T> createMessage(const std::string &type, const void *data, size_t len)
{
    return static_unique_ptr_cast<T, ::google::protobuf::Message>(createMessage(type, data, len));
}

/**
 * Create the protobuf message from a coded input stream. The stream is expected to contains first a
 * varint of length and followed by that length of bytes as the message.
 *
 * @return created Message, or nullptr if specified type not found or data is malformatted.
 */
ProtoPtr createLenLimitedMessage(const std::string &type, ::google::protobuf::io::CodedInputStream *stream);

template<typename T>
std::unique_ptr<T> createLenLimitedMessage(const std::string &type,
                                           ::google::protobuf::io::CodedInputStream *stream)
{
    return static_unique_ptr_cast<T, ::google::protobuf::Message>(createLenLimitedMessage(type, stream));
}

/**
 * Create an empty message object of specified type name `type`.
 *
 * @return created Message, or nullptr if not found.
 */
ProtoPtr newMessage(const std::string &type);
} // namespace symbiotic::salus

#endif // PROTOUTILS_H
