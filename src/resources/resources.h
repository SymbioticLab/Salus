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

#ifndef SALUS_EXEC_RESOURCES_H
#define SALUS_EXEC_RESOURCES_H

#include "../execution/devices.h"
#include "../utils/macros.h"
#include "../utils/pointerutils.h"
#include "../platform/thread_annotations.h"

#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>

enum class ResourceType
{
    COMPUTE,
    MEMORY,
    GPU_STREAM,
    /**
     * @brief A special type of resource that has only 1 unit of it.
     */
    EXCLUSIVE,

    UNKNOWN = 1000,
};

std::string enumToString(const ResourceType &rt);
ResourceType resourceTypeFromString(const std::string &rt);

struct ResourceTag
{
    ResourceType type;
    salus::DeviceSpec device;

    static ResourceTag fromString(const std::string &str);

    std::string DebugString() const;

private:
    friend bool operator==(const ResourceTag &lhs, const ResourceTag &rhs);
    friend bool operator!=(const ResourceTag &lhs, const ResourceTag &rhs);

    auto tie() const
    {
        return std::tie(type, device);
    }
};

inline bool operator==(const ResourceTag &lhs, const ResourceTag &rhs)
{
    return lhs.tie() == rhs.tie();
}

inline bool operator!=(const ResourceTag &lhs, const ResourceTag &rhs)
{
    return lhs.tie() != rhs.tie();
}

namespace std {
template<>
class hash<ResourceTag>
{
public:
    inline size_t operator()(const ResourceTag &tag) const
    {
        size_t res = 0;
        sstl::hash_combine(res, tag.type);
        sstl::hash_combine(res, tag.device);
        return res;
    }
};
} // namespace std

using Resources = std::unordered_map<ResourceTag, size_t>;

namespace resources {
/**
 * @brief Whether 'avail' contains 'req'
 * @param avail
 * @param req
 * @return true iff 'avail' contains 'req'
 */
bool contains(const Resources &avail, const Resources &req);

/**
 * @brief Whether 'lhs' contains all resource types in 'rhs'
 *
 * @param lhs
 * @param rhs
 * @return
 */
bool compatible(const Resources &lhs, const Resources &rhs);

/**
 * @brief Remove resource types with non-positive capacity
 * @param lhs
 * @return
 */
Resources &removeInvalid(Resources &lhs);

/**
 * @brief Merge 'rhs' into 'lhs'
 *
 * @param lhs
 * @param rhs
 * @param skipNonExist Whether to skip resource types that only present in 'rhs'
 * @return reference to 'lhs'
 */
Resources &merge(Resources &lhs, const Resources &rhs, bool skipNonExist = false);

/**
 * @brief Subtract 'rhs' from 'lhs'
 *
 * @param lhs
 * @param rhs
 * @param skipNonExist Whether to skip resource types that only present in 'rhs'
 * @return reference to 'lhs'
 */
Resources &subtract(Resources &lhs, const Resources &rhs, bool skipNonExist = false);

/**
 * @brief Subtract 'rhs' from 'lhs', anything not contained in 'lhs' is skipped
 * @param lhs
 * @param rhs
 * @return actual res subtracted
 */
Resources subtractBounded(Resources &lhs, const Resources &rhs);

/**
 * @brief Multiply a scale to every resource type in 'lhs'
 * @param lhs
 * @param scale
 * @return reference to 'lhs'
 */
Resources &scale(Resources &lhs, double scale);

std::string DebugString(const Resources &res, const std::string &indent = "");

// some handy constant
constexpr ResourceTag CPU0Memory {ResourceType::MEMORY, salus::devices::CPU0};
constexpr ResourceTag GPU0Memory {ResourceType::MEMORY, salus::devices::GPU0};
constexpr ResourceTag GPU1Memory {ResourceType::MEMORY, salus::devices::GPU1};
} // namespace resources

inline std::ostream& operator<<(std::ostream& out, const Resources& res)
{
    return out << resources::DebugString(res);
}

struct ResStats
{
    /**
     * @brief Peak usage excluding persist
     */
    size_t temporary = 0;
    /**
     * @brief Persist usage across iterations
     */
    size_t persist = 0;

    /**
     * @brief Number of allocations per iter
     */
    size_t count = 0;

    std::string DebugString() const;
};
/**
 * @brief Tracks iterations' allocation, making sure no two iterations with
 * risk allocations can run at the same time.
 */
class AllocationRegulator
{
public:
    struct Ticket {
        uint64_t as_int;

        Ticket() = default;
        constexpr Ticket(uint64_t v, AllocationRegulator *r) : as_int(v), reg(r) {}
        Ticket(const Ticket &) = default;

        Ticket &operator=(const Ticket &) = default;
        Ticket &operator=(uint64_t v) { as_int = v; return *this; }

        bool operator==(const Ticket &rhs) const noexcept { return as_int == rhs.as_int; }
        bool operator!=(const Ticket &rhs) const noexcept { return !(*this == rhs); }

        constexpr operator bool() const { return as_int != 0; }

        /**
         * @brief Start an allocation phase using a ticket. Making sure
         * at least 'cap' resource is available or will be available
         * for allocation without deadlocking
         * @param cap
         * @return the allocation ticket, or an empty optional if the request
         * can not be fulfilled
         */
        bool beginAllocation(const Resources &res);

        /**
         * @brief Stop current allocation phase, releasing 'res' amount of holding resources.
         *
         * Note that the ticket is still
         * @param res
         * @return
         */
        void endAllocation(const Resources &res);

        /**
         * @brief Finish the use of the ticket, releasing any remaining resources
         * associated with the ticket.
         * @param ticket
         */
        void finishJob();

        std::string DebugString() const
        {
            return reg->DebugString();
        }

    private:
        AllocationRegulator *reg;
    };

    AllocationRegulator();
    // Read limits from hardware, and capped by cap
    explicit AllocationRegulator(const Resources &cap);

    ~AllocationRegulator() = default;

    /**
     * @brief Register and get a ticket that can be used to start
     * allocation phases
     */
    Ticket registerJob();

    std::string DebugString() const;

private:
    mutable std::mutex m_mu;

    uint64_t m_next = 0 GUARDED_BY(m_mu);

    Resources m_limits GUARDED_BY(m_mu);

    struct JobState
    {
        Resources inuse;
    };

    struct TicketHasher
    {
        size_t operator() (Ticket t) const noexcept
        {
            using std::hash;
            return hash<uint64_t>{}(t.as_int);
        }
    };

    std::unordered_map<Ticket, JobState, TicketHasher> m_jobs GUARDED_BY(m_mu);
};

/**
 * A monitor of resources. This class is thread-safe.
 */
class ResourceMonitor
{
public:
    ResourceMonitor() = default;

    /**
     * @brief Read limits from hardware
     */
    void initializeLimits();
    /**
     * @brief Read limits from hardware, and capped by cap
     */
    void initializeLimits(const Resources &cap);

    /**
     * @brief Try pre-allocate resources
     * @param iterTicket making sure you get an iteration allocation ticket
     * @param req Requested resources to pre-allocate
     * @param missing If not null, contains missing resources that would have make the allocation succeed. Ignored when
     * the allocation succeed.
     * @return An ticket when the pre-allocation succeed, otherwise empty.
     */
    std::optional<uint64_t> preAllocate(const Resources &req, Resources *missing);

    // Allocate resources from pre-allocated resources, if res < reserved, gauranteed to succeed
    // otherwise may return false
    bool allocate(uint64_t ticket, const Resources &res);

    /**
     * @brief Releases remaining pre-allocated resources from ticket `ticket`.
     */
    void freeStaging(uint64_t ticket);

    /**
     * @brief Frees resources `res` for ticket `ticket`.
     * @returns true if the ticket holds no more resources.
     */
    bool free(uint64_t ticket, const Resources &res);

    std::vector<std::pair<size_t, uint64_t>> sortVictim(const std::unordered_set<uint64_t> &candidates) const;

    Resources queryUsages(const std::unordered_set<uint64_t> &tickets) const;

    std::optional<Resources> queryUsage(uint64_t ticket) const;
    bool hasUsage(uint64_t ticket) const;

    struct LockedProxy
    {
        SALUS_DISALLOW_COPY_AND_ASSIGN(LockedProxy);

        explicit LockedProxy(sstl::not_null<ResourceMonitor*> resMon)
            : m_resMonitor(resMon)
        {
            m_resMonitor->m_mu.lock();
        }

        LockedProxy(LockedProxy &&other) noexcept
            : m_resMonitor(other.m_resMonitor)
        {
            other.m_resMonitor = nullptr;
        }

        LockedProxy &operator=(LockedProxy &&other) noexcept
        {
            release();
            using std::swap;
            swap(m_resMonitor, other.m_resMonitor);
            return *this;
        }

        ~LockedProxy()
        {
            release();
        }

        bool allocate(uint64_t ticket, const Resources &res);
        bool free(uint64_t ticket, const Resources &res);
        std::optional<Resources> queryStaging(uint64_t ticket) const;

    private:
        void release()
        {
            if (m_resMonitor) {
                m_resMonitor->m_mu.unlock();
                m_resMonitor = nullptr;
            }
        }

        ResourceMonitor *m_resMonitor;
    };

    LockedProxy lock()
    {
        return LockedProxy(this);
    }

    std::string DebugString() const;

private:
    bool allocateUnsafe(uint64_t ticket, const Resources &res);
    bool freeUnsafe(uint64_t ticket, const Resources &res);
    std::optional<Resources> queryStagingUnsafe(uint64_t ticket) const;

    mutable std::mutex m_mu;

    // 0 is invalid ticket
    uint64_t m_nextTicket = 1;

    /**
     * @brief Available resources
     */
    Resources m_limits;

    /**
     * @brief Staging resources
     */
    std::unordered_map<uint64_t, Resources> m_staging;

    /**
     * @brief In-use resources
     */
    std::unordered_map<uint64_t, Resources> m_using;
};

#endif // SALUS_EXEC_RESOURCES_H
