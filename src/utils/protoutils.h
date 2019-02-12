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

#ifndef SALUS_SSTL_PROTOUTILS_H
#define SALUS_SSTL_PROTOUTILS_H

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

namespace sstl {
/**
 * @brief Create the protobuf message of specific type name `type` from a byte buffer `data` of length `len`.
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
 * @brief Create the protobuf message from a coded input stream. The stream is expected to contains first a
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
} // namespace sstl

#endif // SALUS_SSTL_PROTOUTILS_H
