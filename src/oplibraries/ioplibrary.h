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

#ifndef IOPLIBRARY_H
#define IOPLIBRARY_H

#include "executor.pb.h"

#include <memory>
#include <unordered_map>

class ITask
{
public:
    virtual executor::ResultCode run() = 0;
    virtual executor::OpContextDef contextDef() = 0;
};

/**
 * @todo write docs
 */
class IOpLibrary
{
public:
    virtual bool accepts(const executor::OpKernelDef &operation) = 0;

    virtual ITask *createTask(const executor::OpKernelDef &opeartion, const executor::OpContextDef &context) = 0;
};

class OpLibraryRegistary
{
public:
    OpLibraryRegistary();

    ~OpLibraryRegistary() = default;

    void registerOpLibrary(executor::OpKernelDef::OpLibraryType libraryType, std::unique_ptr<IOpLibrary> &&library);
    IOpLibrary *findSuitableOpLibrary(const executor::OpKernelDef &opdef) const;

    static OpLibraryRegistary &instance();

private:
    std::unordered_map<executor::OpKernelDef::OpLibraryType, std::unique_ptr<IOpLibrary>> m_opLibraries;
};

#endif // IOPLIBRARY_H
