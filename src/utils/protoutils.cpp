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

#include "protoutils.h"

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

namespace sstl {

ProtoPtr newMessage(const std::string &type)
{
    auto desc = protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(type);
    if (!desc) {
//        LOG(ERROR) << "Protobuf descriptor not found for type name: " << type;
        return {};
    }

    auto message = protobuf::MessageFactory::generated_factory()->GetPrototype(desc)->New();
    if (!message) {
//        LOG(ERROR) << "Failed to create message object from descriptor of type name: " << type;
        return {};
    }

    return ProtoPtr(message);
}

ProtoPtr createMessage(const std::string &type, const void *data, size_t len)
{
    auto message = newMessage(type);
    if (!message) {
        return {};
    }

    auto ok = message->ParseFromArray(data, len);
    if (!ok) {
//        LOG(ERROR) << "Failed to parse data buffer of length " << len << " as proto message: " << type;
        return {};
    }

    return message;
}

ProtoPtr createLenLimitedMessage(const std::string &type, protobuf::io::CodedInputStream *stream)
{
    auto limit = stream->ReadLengthAndPushLimit();
    auto msg = newMessage(type);
    if (!msg) {
        return {};
    }
    if (!(msg->ParseFromCodedStream(stream) && stream->ConsumedEntireMessage())) {
//        LOG(ERROR) << "Malformatted message received in CodedInputStream for type: " << type;
        return {};
    }
    stream->PopLimit(limit);

    return msg;
}

} // namespace sstl
