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

#ifndef SALUS_EXEC_RESOURCECONTEXT_H
#define SALUS_EXEC_RESOURCECONTEXT_H

#include "resources/resources.h"
#include "config.h"

#include <boost/container/small_vector.hpp>

#include <atomic>

struct SessionItem;
using PSessionItem = std::shared_ptr<SessionItem>;

namespace salus {
class AllocationListener;
/**
 * @brief Main interface for exectask to allocate resources
 */
class ResourceContext
{
    ResourceMonitor &resMon;
    const uint64_t m_graphId;
    const DeviceSpec m_spec;

    const uint64_t m_ticket;

    std::atomic<bool> m_hasStaging;

    boost::container::small_vector<std::shared_ptr<AllocationListener>, 2> m_listeners;

    friend class TaskExecutor;

    void releaseStaging();

public:

#if defined(SALUS_ENABLE_STATIC_STREAM)
    std::string sessHandle{};
#endif
    /**
     * @brief Construct a resource context
     * @param item
     * @param resMon
     */
    ResourceContext(ResourceMonitor &resMon, uint64_t graphId, const DeviceSpec &spec, uint64_t ticket);

    ResourceContext(const ResourceContext &other, const DeviceSpec &spec);

    const salus::DeviceSpec &spec() const
    {
        return m_spec;
    }
    uint64_t ticket() const
    {
        return m_ticket;
    }

    ~ResourceContext();

    /**
     * @brief Add a listener for allocation. This is *not* thread safe.
     * @param l
     */
    void addListener(std::shared_ptr<AllocationListener> &&l)
    {
        m_listeners.emplace_back(std::move(l));
    }

    struct OperationScope
    {
        explicit OperationScope(const ResourceContext &context, ResourceMonitor::LockedProxy &&proxy)
            : valid(false)
            , proxy(std::move(proxy))
            , res()
            , context(context)
        {
        }

        OperationScope(OperationScope &&scope) noexcept
            : valid(scope.valid)
            , proxy(std::move(scope.proxy))
            , res(std::move(scope.res))
            , context(scope.context)
        {
            scope.valid = false;
        }

        ~OperationScope()
        {
            commit();
        }

        operator bool() const // NOLINT
        {
            return valid;
        }

        void rollback();

        const Resources &resources() const
        {
            return res;
        }

    private:
        void commit();

        friend class ResourceContext;

        bool valid;
        ResourceMonitor::LockedProxy proxy;
        Resources res;
        const ResourceContext &context;
    };

    /**
     * @brief Allocate all resource of type `type' in staging area
     * @param type
     * @return
     */
    OperationScope alloc(ResourceType type) const;

    OperationScope alloc(ResourceType type, size_t num) const;

    void dealloc(ResourceType type, size_t num) const;

    void removeTicketFromSession() const;
};

std::ostream &operator<<(std::ostream &os, const ResourceContext &c);

} // namespace salus

#endif // SALUS_EXEC_RESOURCECONTEXT_H
