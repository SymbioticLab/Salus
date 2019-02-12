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
