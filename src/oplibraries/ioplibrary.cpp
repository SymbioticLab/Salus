/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017 Peifeng Yu <peifeng@umich.edu>
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
    utils::Guard g(m_mu);
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
