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

#ifndef SALUS_OPLIB_TENSORFLOW_TFINSTANCE_H
#define SALUS_OPLIB_TENSORFLOW_TFINSTANCE_H

#include "oplibraries/tensorflow/tfoplibraryv2.h"
#include "oplibraries/tensorflow/tfutils.h"
#include "platform/thread_annotations.h"
#include "utils/cpp17.h"
#include "utils/macros.h"
#include "utils/pointerutils.h"

#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tensorflow {
class ConfigProto;
class Device;
class DeviceMgr;
class Env;
} // namespace tensorflow

namespace salus::oplib::tensorflow {

class TFSession;
struct HandlerCallback;
class LaneMgr;

/**
 * @brief Represents the tensorflow instance used in Salus.
 */
class TFInstance
{
    sstl::not_null<tf::Env *> m_env;

    friend class TFSession;
    std::mutex m_mu;
    std::unordered_map<std::string, std::shared_ptr<TFSession>> m_sessions GUARDED_BY(m_mu);

    struct DeviceContainer
    {
        static constexpr int MaxDeviceType = 2;
        static constexpr int MaxDeviceId = 3;

        tf::Device *specToTF[MaxDeviceType][MaxDeviceId] = {};
        std::unique_ptr<tf::DeviceMgr> deviceMgr;
        // devices in m_devices owned by m_deviceMgr
        std::vector<tf::Device *> devices;

        DeviceContainer();

        static constexpr std::string_view SpecToTFDevName(const DeviceSpec &spec)
        {
            using namespace std::literals::string_view_literals;

            static_assert(sstl::to_underlying(DeviceType::CPU) == 0,
                          "The order of dtstrings depends on the order of DeviceType enum");
            static_assert(sstl::to_underlying(DeviceType::GPU) == 1,
                          "The order of dtstrings depends on the order of DeviceType enum");
            // NOTE: this must match the order of DeviceType enum
            // clang-format off
            constexpr std::string_view tfdevnames[MaxDeviceType][MaxDeviceId] = {
                { "CPU:0"sv, "CPU:1"sv, "CPU:2"sv, },
                { "GPU:0"sv, "GPU:1"sv, "GPU:2"sv, },
            };
            // clang-format on
            return tfdevnames[sstl::to_underlying(spec.type)][spec.id];
        }

        SALUS_DISALLOW_COPY_AND_ASSIGN(DeviceContainer);
    };
    const DeviceContainer m_devCon;

    std::unique_ptr<LaneMgr> m_laneMgr;

public:
    SALUS_DISALLOW_COPY_AND_ASSIGN(TFInstance);

    TFInstance();
    ~TFInstance();

    static TFInstance &instance();

    static auto namePrefix()
    {
        return "/job:salus/replica:0/task:0";
    }

    auto &env() const
    {
        return *m_env;
    }

#if !defined(SALUS_ENABLE_SIEXECUTOR)
    tf::Device *tfdevice(const DeviceSpec &spec) const;
#endif

    /**
     * @brief find session
     */
    std::shared_ptr<TFSession> findSession(const std::string &sessHandle);

    /**
     * @brief find session and remove it from storage
     */
    std::shared_ptr<TFSession> popSession(const std::string &sessHandle);

#define DECLARE_HANDLER(name)                                                                                          \
    void handle##name(std::unique_ptr<tf::name##Request> &&req, tf::name##Response &resp, HandlerCallback &&cb)

    DECLARE_HANDLER(CreateSession);
    DECLARE_HANDLER(CloseSession);
    DECLARE_HANDLER(ListDevices);
    DECLARE_HANDLER(Reset);

#undef DECLARE_HANDLER

    /**
     * @brief for debugging, dump memory map for GPU
     */
    std::string maybeDumpGPUMemoryMap(tf::Device *dev) const;
};

} // namespace salus::oplib::tensorflow

#endif // SALUS_OPLIB_TENSORFLOW_TFINSTANCE_H
