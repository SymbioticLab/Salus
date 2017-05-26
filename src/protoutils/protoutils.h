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

#include <google/protobuf/message.h>

#include <memory>

typedef std::unique_ptr<::google::protobuf::Message> ProtoPtr;

namespace protoutils {
/**
 * Create the protobuf message of specific type name `type` from a byte buffer `data` of length `len`.
 *
 * @return created Message, or nullptr if specified type not found.
 */
ProtoPtr createMessage(const std::string type, const void *data, size_t len);
}

#endif // PROTOUTILS_H
