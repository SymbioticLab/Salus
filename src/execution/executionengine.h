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

#ifndef EXECUTIONENGINE_H
#define EXECUTIONENGINE_H

#include "devices.h"

#include "oplibraries/ioplibrary.h"
#include "platform/logging.h"

#include <q/lib.hpp>
#include <q/promise.hpp>
#include <q/execution_context.hpp>
#include <q/threadpool.hpp>

#include <memory>

/**
 * @todo write docs
 */
class ExecutionEngine
{
public:
    static ExecutionEngine &instance();

    ~ExecutionEngine();

    template<typename ResponseType>
    q::promise<std::unique_ptr<ResponseType>> enqueue(PTask &&task)
    {
        using PResponse = std::unique_ptr<ResponseType>;

        return q::make_promise_of<PResponse>(m_qec->queue(),
                                             [this, task = std::move(task)](auto resolve,
                                                                            auto reject){
            try {
                if (this->schedule(task.get())) {
                    if (task->isAsync()) {
                        task->runAsync<ResponseType>([resolve](PResponse &&ptr){
                            resolve(std::move(ptr));
                        });
                    } else {
                        resolve(task->run<ResponseType>());
                    }
                } else {
                    reject(std::logic_error("Task failed to prepare"));
                }
            } catch (std::exception &err) {
                reject(err);
            }
        });
    }

    template<typename ResponseType>
    q::promise<std::unique_ptr<ResponseType>> emptyPromise()
    {
        using PResponse = std::unique_ptr<ResponseType>;
        return q::with(m_qec->queue(), PResponse(nullptr));
    }

    template<typename ResponseType>
    q::promise<ResponseType> makePromise(ResponseType &&t)
    {
        return q::with(m_qec->queue(), std::move(t));
    }

    template<typename ResponseType>
    q::promise<ResponseType> makePromise(const ResponseType &t)
    {
        return q::with(m_qec->queue(), t);
    }

protected:
    bool schedule(ITask *t);

private:
    ExecutionEngine();

    using qExecutionContext = q::specific_execution_context_ptr<q::threadpool>;

    q::scope m_qscope;
    qExecutionContext m_qec;
};

#endif // EXECUTIONENGINE_H
