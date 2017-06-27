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
#include "utils/macros.h"

ITask::~ITask() = default;

bool ITask::prepare(DeviceType &dev)
{
    UNUSED(dev);

    return true;
}

bool ITask::isAsync()
{
    return false;
}

void ITask::runAsync(DoneCallback cb)
{
    UNUSED(cb);
    ERR("Not Implemented");
}

IOpLibrary::~IOpLibrary() = default;

OpLibraryRegistary &OpLibraryRegistary::instance()
{
    static OpLibraryRegistary registary;
    return registary;
}

OpLibraryRegistary::OpLibraryRegistary() = default;

OpLibraryRegistary::Register::Register(executor::OpLibraryType libraryType,
                                       std::unique_ptr<IOpLibrary> &&library)
{
    OpLibraryRegistary::instance().registerOpLibrary(libraryType, std::move(library));
}

void OpLibraryRegistary::registerOpLibrary(executor::OpLibraryType libraryType,
                                           std::unique_ptr<IOpLibrary> &&library)
{
    m_opLibraries[libraryType] = std::move(library);
}

IOpLibrary *OpLibraryRegistary::findOpLibrary(const executor::OpLibraryType libraryType) const
{
    if (m_opLibraries.count(libraryType) <= 0) {
        WARN("No OpLibrary registered under the library type {}",
             executor::OpLibraryType_Name(libraryType));
        return nullptr;
    }
    return m_opLibraries.at(libraryType).get();
}

IOpLibrary * OpLibraryRegistary::findSuitableOpLibrary(const executor::OpKernelDef& opdef) const
{
    for (const auto &elem : m_opLibraries) {
        if (elem.first == opdef.oplibrary() && elem.second->accepts(opdef)) {
            return elem.second.get();
        }
    }
    WARN("No suitable OpLibrary found for {}", opdef);
    return nullptr;
}
