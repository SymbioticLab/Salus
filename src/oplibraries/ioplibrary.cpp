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

#include "ioplibrary.h"

#include "platform/logging.h"
#include "utils/threadutils.h"

IOpLibrary::~IOpLibrary() = default;

OpLibraryRegistary &OpLibraryRegistary::instance()
{
    static OpLibraryRegistary registary;
    return registary;
}

OpLibraryRegistary::OpLibraryRegistary() = default;

OpLibraryRegistary::~OpLibraryRegistary() = default;

OpLibraryRegistary::Register::Register(executor::OpLibraryType libraryType,
                                       std::unique_ptr<IOpLibrary> &&library,
                                       int priority)
{
    OpLibraryRegistary::instance().registerOpLibrary(libraryType, std::move(library), priority);
}

void OpLibraryRegistary::registerOpLibrary(executor::OpLibraryType libraryType,
                                           std::unique_ptr<IOpLibrary> &&library, int priority)
{
    std::lock_guard<std::mutex> guard(m_mu);
    auto iter = m_opLibraries.find(libraryType);
    if (iter == m_opLibraries.end()) {
        m_opLibraries[libraryType] = {std::move(library), priority};
    } else {
        if (iter->second.priority < priority) {
            iter->second = {std::move(library), priority};
        } else if (iter->second.priority == priority) {
            LOG(FATAL) << "Duplicate registration of device factory for type "
                       << executor::OpLibraryType_Name(libraryType)
                       << " with the same priority " << priority;
        }
    }
}

void OpLibraryRegistary::initializeLibraries()
{
    auto g = sstl::with_guard(m_mu);
    ++initialized;
    if (initialized > 1) {
        return;
    }

    auto it = m_opLibraries.begin();
    auto itend = m_opLibraries.end();
    while (it != itend) {
        if (!it->second.library->initialize()) {
            LOG(ERROR) << "Removing OpLibrary from registary that failed to initialize: " << as_hex(it->second.library);
            it = m_opLibraries.erase(it);
        } else {
            ++it;
        }
    }
}

void OpLibraryRegistary::uninitializeLibraries()
{
    auto g = sstl::with_guard(m_mu);

    --initialized;
    if (initialized <= 0) {
        return;
    }

    for (auto &p : m_opLibraries) {
        p.second.library->uninitialize();
    }
}

IOpLibrary *OpLibraryRegistary::findOpLibrary(const executor::OpLibraryType libraryType) const
{
    std::lock_guard<std::mutex> guard(m_mu);
    auto iter = m_opLibraries.find(libraryType);
    if (iter == m_opLibraries.end()) {
        LOG(ERROR) << "No OpLibrary registered under the library type "
                   << executor::OpLibraryType_Name(libraryType);
        return nullptr;
    }
    return iter->second.library.get();
}

IOpLibrary * OpLibraryRegistary::findSuitableOpLibrary(const executor::OpKernelDef& opdef) const
{
    for (const auto &elem : m_opLibraries) {
        if (elem.first == opdef.oplibrary() && elem.second.library->accepts(opdef)) {
            return elem.second.library.get();
        }
    }
    LOG(ERROR) << "No suitable OpLibrary found for " << opdef;
    return nullptr;
}
