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

#ifndef SALUS_SSTL_OBJECTPOOL_H
#define SALUS_SSTL_OBJECTPOOL_H

#include <concurrentqueue.h>

#include <memory>

namespace sstl {
/**
 * An pool of objects. This is thread safe.
 */
template<typename T>
class ObjectPool : public std::enable_shared_from_this<ObjectPool<T>>
{
private:
    struct ExternalDeleter
    {
        explicit ExternalDeleter(std::weak_ptr<ObjectPool<T>> &&pool) noexcept
            : m_pool(std::move(pool))
        {
        }

        void operator()(T *ptr) noexcept
        {
            if (auto pool_ptr = m_pool.lock()) {
                pool_ptr->add(std::unique_ptr<T>{ptr});
            } else {
                // object pool goes out of scope before the object
                std::default_delete<T>{}(ptr);
            }
        }

    private:
        std::weak_ptr<ObjectPool<T>> m_pool;
    };

public:
    using ptr_type = std::unique_ptr<T, ExternalDeleter>;

    ObjectPool() noexcept
        : m_frees(std::thread::hardware_concurrency(), 0, std::thread::hardware_concurrency())
        , m_token(m_frees)
    {
    }

    void add(std::unique_ptr<T> &&t) noexcept
    {
        m_frees.enqueue(std::move(t));
    }

    template<typename... Args>
    ptr_type acquire(Args && ... args) noexcept
    {
        std::unique_ptr<T> tmp;
        if (m_frees.try_dequeue(m_token, tmp)) {
            tmp->reset(std::forward<Args>(args)...);
        } else {
            tmp = std::make_unique<T>(std::forward<Args>(args)...);
        }

        // NOTE: 'this->' is needed before weak_from_this because of template 2 pheases name lookup
        // see https://stackoverflow.com/a/15531940/2441376
        return ptr_type{tmp.release(), ExternalDeleter{this->weak_from_this()}};
    }

    size_t size() const noexcept
    {
        return m_frees.size_approx();
    }

private:
    using value_type = std::unique_ptr<T>;
    using FreeList = moodycamel::ConcurrentQueue<value_type>;
    using Token = moodycamel::ConsumerToken;
    FreeList m_frees;
    Token m_token;
};

} // namespace sstl

#endif // SALUS_SSTL_OBJECTPOOL_H
