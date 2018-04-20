//
// Created by peifeng on 4/19/18.
//

#ifndef SALUS_EXEC_ALLOCATIONLISTENER_H
#define SALUS_EXEC_ALLOCATIONLISTENER_H

#include "resources/resources.h"

namespace salus {
class AllocationListener
{
public:
    virtual ~AllocationListener() = default;
    /**
     * @brief Should be called by ResourceMonitor
     * @param ticket
     * @param res
     */
    virtual void notifyAlloc(const std::string &graphId, uint64_t ticket, const ResourceTag &rt, size_t num) = 0;

    /**
     * @brief Called when dealloc happens
     * @param ticket the ticket used
     * @param rt resource tag
     * @param num amount
     * @param last if this was the last allocation from ticket
     */
    virtual void notifyDealloc(const std::string &graphId, uint64_t ticket, const ResourceTag &rt, size_t num, bool last) = 0;
};

} // namespace salus

#endif // SALUS_EXEC_ALLOCATIONLISTENER_H
