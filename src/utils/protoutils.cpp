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

#include "protoutils.h"

#include "platform/logging.h"

#ifndef NDEBUG
#define NDEBUG
#define NEED_UNDEF_NDEBUG
#endif

#include <google/protobuf/io/coded_stream.h>

#ifdef NEED_UNDEF_NDEBUG
#undef NDEBUG
#undef NEED_UNDEF_NDEBUG
#endif

namespace protobuf = ::google::protobuf;

ProtoPtr utils::newMessage(const std::string &type)
{
    auto desc = protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(type);
    if (!desc) {
        WARN("Protobuf descriptor not found for type name: {}", type);
        return {};
    }

    auto message = protobuf::MessageFactory::generated_factory()->GetPrototype(desc)->New();
    if (!message) {
        WARN("Failed to create message object from descriptor of type name: {}", type);
        return {};
    }

    return ProtoPtr(message);
}

ProtoPtr utils::createMessage(const std::string &type, const void* data, size_t len)
{
    auto message = newMessage(type);
    if (!message) {
        return {};
    }

    auto ok = message->ParseFromArray(data, len);
    if (!ok) {
        WARN("Failed to parse data buffer of length {} as proto message: {}", len, type);
        return {};
    }

    return message;
}

ProtoPtr utils::createLenLimitedMessage(const std::string &type, protobuf::io::CodedInputStream *stream)
{
    auto limit = stream->ReadLengthAndPushLimit();
    auto msg = utils::newMessage(type);
    if (!msg) {
        return {};
    }
    if (!(msg->ParseFromCodedStream(stream) && stream->ConsumedEntireMessage())) {
        WARN("Malformatted message received in CodedInputStream for type: {}", type);
        return {};
    }
    stream->PopLimit(limit);

    return msg;
}
