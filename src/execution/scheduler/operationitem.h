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
 */

#ifndef SALUS_EXEC_OPERATIONITEM_H
#define SALUS_EXEC_OPERATIONITEM_H

#include <memory>
#include <chrono>

class OperationTask;
struct SessionItem;
struct OperationItem
{
    std::weak_ptr<SessionItem> sess;
    std::unique_ptr<OperationTask> op;

    size_t hash() const
    {
        return reinterpret_cast<size_t>(this);
    }

private:
    friend class ExecutionEngine;
};
using POpItem = std::shared_ptr<OperationItem>;

#endif // SALUS_EXEC_OPERATIONITEM_H
