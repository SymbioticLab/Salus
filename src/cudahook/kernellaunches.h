/*
 * Copyright (c) 2019, peifeng <email>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SALUS_KERNELLAUNCHES_H
#define SALUS_KERNELLAUNCHES_H

#include <unordered_map>
#include <cstdint>

namespace salus {

constexpr auto KernelLaunchCallbackFuncationName = "salus_kernel_launch_callback";

using FnKernelLaunchCallback = void (unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                     unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                     unsigned int sharedMemBytes, void *hStream);

namespace details {

struct KernelParams
{
    uint32_t gridX = 0;
    uint32_t gridY = 0;
    uint32_t gridZ = 0;
    uint32_t blkX = 0;
    uint32_t blkY = 0;
    uint32_t blkZ = 0;
    uint32_t shdMem = 0;
    void *stream = nullptr;
};

} // namespace details

class DetectorCuLaunchKernel
{
    static FnKernelLaunchCallback *m_callback;

public:
    static void setCallback(FnKernelLaunchCallback *callback)
    {
        m_callback = callback;
    }

    static void installHooks();

    static DetectorCuLaunchKernel &localInstance();

    /*
     * Actual detector logic below
     */
private:
    enum class State {
        Idle, Found
    };
    State m_state = State::Idle;

    details::KernelParams m_kernelParams;

    void fire();

public:
    DetectorCuLaunchKernel() = default;

    void onCuLaunchKernel(details::KernelParams params);
};

class DetectorCuLaunch
{
    static FnKernelLaunchCallback *m_callback;
public:
    static void setCallback(FnKernelLaunchCallback *callback)
    {
        m_callback = callback;
    }

    static void installHooks();

    static DetectorCuLaunch &localInstance();

    // Actual detector logic below
private:
    enum class State {
        Idle
    };
    State m_state = State::Idle;

    std::unordered_map<void*, details::KernelParams> m_params;

    details::KernelParams &ensureParams(void *func);

    void fire(const details::KernelParams &params);

public:
    void onCuFuncSetBlockShape(void* f, int x, int y, int z);
    void onCuFuncSetSharedSize(void* f, unsigned int bytes);
    void onCuLaunch(void *func);
    void onCuLaunchGrid(void* f, int grid_width, int grid_height);
    void onCuLaunchGridAsync(void* f, int grid_width, int grid_height, void* stream);
};

class KernelLaunches
{
    void *m_selfHandle = nullptr;
    FnKernelLaunchCallback *m_kernelLaunchCallback = nullptr;

    bool m_debugging = false;

public:
    KernelLaunches() noexcept;

    bool debugging() const { return m_debugging; }
};

} // namespace salus

#endif // SALUS_KERNELLAUNCHES_H
