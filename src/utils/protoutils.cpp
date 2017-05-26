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

ProtoPtr utils::createMessage(const std::string type, const void* data, size_t len)
{
    auto desc = ::google::protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(type);
    if (!desc) {
        WARN("Protobuf descriptor not found for type name: {}", type);
        return {};
    }

    auto message = ::google::protobuf::MessageFactory::generated_factory()->GetPrototype(desc)->New();
    if (!message) {
        WARN("Failed to create message object from descriptor of type name: {}", type);
        return {};
    }

    auto ok = message->ParseFromArray(data, len);
    if (!ok) {
        WARN("Failed to parse data buffer of length {} as proto message: {}", len, type);
        return {};
    }

    return ProtoPtr(message);
}
