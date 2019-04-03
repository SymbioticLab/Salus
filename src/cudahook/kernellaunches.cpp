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

#include "kernellaunches.h"

#include "realdlsym.h"
#include "cudahook.h"

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>

namespace {
} // namespace

namespace salus {

KernelLaunches kl [[maybe_unused]];

KernelLaunches::KernelLaunches() noexcept
{
    // get callback function
    m_selfHandle = dlopen(nullptr, RTLD_LAZY);
    if (!m_selfHandle) {
        std::cerr << "Error to get handle to self executable: " << dlerror() << std::endl;
        std::exit(-4);
    }
    m_kernelLaunchCallback = func_cast<FnKernelLaunchCallback*>(real_dlsym(m_selfHandle, KernelLaunchCallbackFuncationName));
    if (!m_kernelLaunchCallback) {
        std::cerr << "Error to find symbol " << KernelLaunchCallbackFuncationName << ": " << dlerror() << std::endl;
        std::exit(-5);
    }
    DetectorCuLaunchKernel::setCallback(m_kernelLaunchCallback);
    DetectorCuLaunch::setCallback(m_kernelLaunchCallback);

    // install hooks to detect kernel launches
    DetectorCuLaunchKernel::installHooks();
    DetectorCuLaunch::installHooks();

    // debug
    auto envDebug = std::getenv("CUDA_HOOK_DEBUG");
    if (envDebug && envDebug[0] == '1') {
        m_debugging = true;
        std::cerr << "CUDA Kernel launch recording started." << std::endl;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// cuLaunchKernel detector
// ---------------------------------------------------------------------------------------------------------------------

FnKernelLaunchCallback *DetectorCuLaunchKernel::m_callback = nullptr;

void DetectorCuLaunchKernel::installHooks()
{
    CudaHook::instance().pre().cuLaunchKernel = [](auto, auto gridX, auto gridY, auto gridZ,
                                                   auto blkX, auto blkY, auto blkZ,
                                                   auto shdMem, auto stream, auto, auto) {
        auto &detector = localInstance();
        detector.onCuLaunchKernel({gridX, gridY, gridZ, blkX, blkY, blkZ, shdMem, stream});
        return 0;
    };
}

DetectorCuLaunchKernel &DetectorCuLaunchKernel::localInstance()
{
    static thread_local DetectorCuLaunchKernel detector;
    return detector;
}

void DetectorCuLaunchKernel::onCuLaunchKernel(details::KernelParams params)
{
    m_kernelParams = params;
    m_state = State::Found;
    fire();
}

void DetectorCuLaunchKernel::fire()
{
    if (m_state != State::Found) {
        return;
    }
    if (m_callback) {
        m_callback(m_kernelParams.gridX, m_kernelParams.gridY, m_kernelParams.gridZ,
                   m_kernelParams.blkX, m_kernelParams.blkY, m_kernelParams.blkZ,
                   m_kernelParams.shdMem, m_kernelParams.stream);
    }
    m_state = State::Idle;
}

// ---------------------------------------------------------------------------------------------------------------------
// cuLaunch detector
// ---------------------------------------------------------------------------------------------------------------------

FnKernelLaunchCallback *DetectorCuLaunch::m_callback = nullptr;

void DetectorCuLaunch::installHooks()
{
    CudaHook::instance().pre().cuFuncSetBlockShape = [](auto f, auto x, auto y, auto z) {
        localInstance().onCuFuncSetBlockShape(f, x, y, z);
        return 0;
    };
    CudaHook::instance().pre().cuFuncSetSharedSize = [](auto f, auto size) {
        localInstance().onCuFuncSetSharedSize(f, size);
        return 0;
    };
    CudaHook::instance().pre().cuLaunch = [](auto f) {
        localInstance().onCuLaunch(f);
        return 0;
    };
    CudaHook::instance().pre().cuLaunchGrid = [](auto f, auto w, auto h) {
        localInstance().onCuLaunchGrid(f, w, h);
        return 0;
    };
    CudaHook::instance().pre().cuLaunchGridAsync = [](auto f, auto w, auto h, auto stream) {
        localInstance().onCuLaunchGridAsync(f, w, h, stream);
        return 0;
    };
}

DetectorCuLaunch &DetectorCuLaunch::localInstance()
{
    static thread_local DetectorCuLaunch detector;
    return detector;
}

void DetectorCuLaunch::onCuFuncSetBlockShape(void *f, int x, int y, int z)
{
    auto &params = ensureParams(f);
    params.blkX = x;
    params.blkY = y;
    params.blkZ = z;
}

void DetectorCuLaunch::onCuFuncSetSharedSize(void *f, unsigned int bytes)
{
    auto &params = ensureParams(f);
    params.shdMem = bytes;
}

void DetectorCuLaunch::onCuLaunch(void *func)
{
    auto &params = ensureParams(func);
    params.gridX = params.gridY = params.gridZ = 1;
    fire(params);
    m_params.erase(func);
}

void DetectorCuLaunch::onCuLaunchGrid(void *f, int grid_width, int grid_height)
{
    auto &params = ensureParams(f);
    params.gridX = grid_width;
    params.gridY = grid_height;
    params.gridZ = 1;
    fire(params);
    m_params.erase(f);
}

void DetectorCuLaunch::onCuLaunchGridAsync(void *f, int grid_width, int grid_height, void *stream)
{
    auto &params = ensureParams(f);
    params.gridX = grid_width;
    params.gridY = grid_height;
    params.gridZ = 1;
    params.stream = stream;
    fire(params);
    m_params.erase(f);
}

details::KernelParams &DetectorCuLaunch::ensureParams(void *func)
{
    auto it = m_params.try_emplace(func, details::KernelParams{1, 1, 1, 1, 1, 1, 0, nullptr}).first;
    return it->second;
}

void DetectorCuLaunch::fire(const details::KernelParams &params)
{
    if (m_callback) {
        m_callback(params.gridX, params.gridY, params.gridZ,
                   params.blkX, params.blkY, params.blkZ,
                   params.shdMem, params.stream);
    }
}

} // namespace salus
