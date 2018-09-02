/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Aetf <aetf@unlimitedcodeworks.xyz>
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

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "oplibraries/tensorflow/tfexception.h"

#include <unordered_map>
#include <sstream>

namespace salus::oplib::tensorflow {

DeviceSpec tfDeviceNameToSpec(const std::string &name)
{

    tf::DeviceNameUtils::ParsedName parsedName;
    if (!tf::DeviceNameUtils::ParseFullName(name, &parsedName)) {
        throw TFException(tf::errors::InvalidArgument("Device name invalid: ", name));
    }
    return DeviceSpec{ tfDeviceTypeToType(parsedName.type), parsedName.id };
}

DeviceType tfDeviceTypeToType(const std::string &type)
{
    return tfDeviceTypeToType(tf::DeviceType(type));
}

DeviceType tfDeviceTypeToType(const tf::DeviceType &type)
{
    static std::unordered_map<std::string, DeviceType> mapping{
        {tf::DEVICE_CPU, DeviceType::CPU},
        {tf::DEVICE_GPU, DeviceType::GPU},
    };

    if (auto it = mapping.find(type.type_string()); it != mapping.end()) {
        return it->second;
    }
    throw TFException(tf::errors::InvalidArgument("Unknown tf::DeviceType: ", type.type()));
}

std::string tfGraphToGraphviz(const tf::Graph &g, const std::string &name)
{
    std::ostringstream os;
    os << "digraph " << name << "{";

    // graph attributes
    os << "graph[num_nodes=" << g.num_nodes()
       << ",num_edges=" << g.num_edges()
       << ","
    << "];";

    // all nodes
    for (auto *n : g.nodes()) {
        os << n->id() << "[name=\"" << n->name() << "\""
           << ",type=\"" << n->type_string() << "\""
           << "];";
    }
    // all edges
    for (auto e : g.edges()) {
        os << e->src()->id() << "->" << e->dst()->id() << "["
           << "src_output=" << e->src_output()
           << ",dst_input=" << e->dst_input()
            << ",id=" << e->id()
           << "];";
    }

    os << "}";
    return os.str();
}

} // namespace salus::oplib::tensorflow
