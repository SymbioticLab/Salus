/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017  Aetf <aetf@unlimitedcodeworks.xyz>
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

#ifndef TFOPLIBRARY_H
#define TFOPLIBRARY_H

#include "ioplibrary.h"

#include <memory>

namespace tensorflow {
class OpKernel;
class OpKernelContext;
}

class TFTask : public ITask
{
public:
    TFTask(tensorflow::OpKernel *kernel, tensorflow::OpKernelContext *context);

    executor::ResultCode run() override;

private:
    std::unique_ptr<tensorflow::OpKernel> m_opkernel;
    std::unique_ptr<tensorflow::OpKernelContext> m_context;
};

/**
 * @todo write docs
 */
class TFOpLibrary : public IOpLibrary
{
public:
    bool accepts(const executor::OpKernelDef &operation) override;
    ITask *createTask(const executor::OpKernelDef &opeartion, const executor::OpContextDef &context) override;
};

#endif // TFOPLIBRARY_H
