//
// Created by aetf on 18-3-30.
//

#ifndef SALUS_SSTL_OBJECTPOOL_H
#define SALUS_SSTL_OBJECTPOOL_H

#include "platform/logging.h"

#include <memory>

namespace sstl {
/**
 * An pool of objects
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
                try {
                    pool_ptr->add(std::unique_ptr<T>{ptr});
                    return;
                } catch (...) {
                }
            }
            std::default_delete<T>{}(ptr);
        }

    private:
        std::weak_ptr<ObjectPool<T>> m_pool;
    };

public:
    using ptr_type = std::unique_ptr<T, ExternalDeleter>;

    void add(std::unique_ptr<T> &&t) noexcept
    {
        m_frees.push(std::move(t));
    }

    template<typename... Args>
    ptr_type acquire(Args... args) noexcept
    {
        T *tmp;
        if (m_frees.empty()) {
            tmp = std::make_unique<T>(std::forward<Args>(args)...).release();
        } else {
            m_frees.top()->reset(std::forward<Args>(args)...);
            tmp = m_frees.top().release();
            m_frees.pop();
        }
        // NOTE: 'this->' is needed before weak_from_this because of template 2 pheases name lookup
        // see https://stackoverflow.com/a/15531940/2441376
        return ptr_type{tmp, ExternalDeleter{this->weak_from_this()}};
    }

    bool empty() const noexcept
    {
        return m_frees.empty();
    }

    size_t size() const noexcept
    {
        return m_frees.size();
    }

private:
    std::stack<std::unique_ptr<T>> m_frees;
};

} // namespace sstl

#endif // SALUS_SSTL_OBJECTPOOL_H
