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

#ifndef SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFINSTANCE_H
#define SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFINSTANCE_H

#include "oplibraries/tensorflow/tfutils.h"
#include "utils/pointerutils.h"

#include <memory>
#include <vector>
#include <exception>

namespace tensorflow {
class ConfigProto;
class Device;
class DeviceMgr;
class Env;
} // namespace tensorflow

namespace symbiotic::salus::oplib::tensorflow {
class TFSession;

/**
 * @brief Represents the tensorflow instance used in Salus.
 */
class TFInstance
{
    utils::not_null<::tensorflow::Env *> m_env;

    std::unique_ptr<::tensorflow::DeviceMgr> m_deviceMgr;
    // devices in m_devices owned by m_deviceMgr
    std::vector<::tensorflow::Device *> m_devices;

    friend class TFSession;

public:
    explicit TFInstance(const ::tensorflow::ConfigProto &config);
    ~TFInstance();

    static auto namePrefix() {
        return "/job:executor/replica:0/task:0";
    }

    auto &env() const { return *m_env; }
    auto &deviceMgr() const { return *m_deviceMgr; }
    auto &devices() const { return m_devices; }

    auto createSession();
};

} // namespace symbiotic::salus::oplib::tensorflow

#endif // SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFINSTANCE_H
