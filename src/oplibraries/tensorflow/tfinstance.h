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
