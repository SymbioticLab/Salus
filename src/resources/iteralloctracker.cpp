//
// Created by peifeng on 4/20/18.
//

#include "resources/iteralloctracker.h"
#include "utils/date.h"
#include "platform/logging.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
using FpSeconds = std::chrono::duration<double, seconds::period>;
using namespace std::chrono_literals;
using namespace date;

namespace salus {

IterAllocTracker::IterAllocTracker(const ResourceTag &tag, size_t window, double peakthr)
    : m_tag(tag)
    , m_peakthr(peakthr)
    , m_window(window)
{
}

bool IterAllocTracker::beginIter(AllocationRegulator::Ticket ticket, ResStats estimation)
{
    CHECK(!m_inIter);

    m_ticket = ticket;
    if (m_numIters == 0) {
        m_est = estimation;
    }

    VLOG(3) << "IterAllocTracker@" << as_hex(this) << "::beginIter ticket=" << m_ticket.as_int
            << ", estimation=" << m_est.DebugString() << ", numIter=" << m_numIters;

    // reset curr
    m_curr = {};
    m_curr.persist = m_est.persist;

    // reset buffer
    m_buf.clear();
    if (m_window == 0 && m_est.count != 0) {
        m_buf.set_capacity(m_est.count / 50);
    } else if (m_window == 0){
        m_buf.set_capacity(50);
    } else {
        m_buf.set_capacity(m_window);
    }

    // reserve res
    Resources cap;
    cap[m_tag] = m_est.temporary;
    VLOG(3) << "IterAllocTracker@" << as_hex(this) << " reserve: " << cap;
    m_inIter = m_ticket.beginAllocation(cap);
    if (m_inIter) {
        ++m_numIters;
    } else {
        VLOG(2) << "Delay iteration due to unsafe resource usage@" << as_hex(this) << ". Ticket: " << m_ticket.as_int << ", Predicted usage: "
                << cap << ", current usage: " << m_ticket.DebugString();
    }
    return m_inIter;
}

bool IterAllocTracker::update(size_t num)
{
    if (num > m_curr.persist) {
        m_curr.temporary = std::max(num - m_curr.persist, m_curr.temporary);
        m_curr.count++;
    }
    VLOG(3) << "IterAllocTracker@" << as_hex(this) << "::update ticket=" << m_ticket.as_int << ", current=" << m_curr.DebugString() << ", numIter=" << m_numIters;

    if (!m_inIter) {
        return false;
    }

    m_buf.push_back({system_clock::now().time_since_epoch().count(), num});

    if (m_buf.size() < 2) {
        return false;
    }

    auto [stx, sty] = m_buf.front();
    auto [edx, edy] = m_buf.back();
    auto slope = (edy - sty) * 1.0 / (edx - stx);
    if (slope < 0 && num >= m_peakthr * m_est.temporary) {
        VLOG(3) << "IterAllocTracker@" << as_hex(this) << "::endAlloc ticket=" << m_ticket.as_int << ", estimation=" << m_est.DebugString() << ", numIter=" << m_numIters;
        m_inIter = false;
        m_ticket.endAllocation({{m_tag, m_est.temporary}});
        return true;
    }

    return false;
}

namespace {
size_t runningAvg(size_t lastAvg, size_t current, int newCount)
{
    if (lastAvg == 0) {
        return current;
    }

    double avg = lastAvg;
    avg = (avg * (newCount - 1) + current) / newCount;
    return static_cast<size_t>(avg);
}
}

void IterAllocTracker::endIter()
{
    CHECK(m_inIter);
    m_inIter = false;

    Resources toRelease{
        {m_tag, m_est.temporary}
    };
    VLOG(3) << "IterAllocTracker@" << as_hex(this) << "::endIter ticket=" << m_ticket.as_int << ", estimation=" << m_est.DebugString()
            << ", numIter=" << m_numIters << ", toRelease=" << toRelease;
    m_ticket.endAllocation(toRelease);
    // update our estimation using running average
    m_est.temporary = runningAvg(m_est.temporary, m_curr.temporary, m_numIters);
    m_est.count = runningAvg(m_est.count, m_curr.count, m_numIters);

    VLOG(3) << "IterAllocTracker@" << as_hex(this) << "::endIter curr=" << m_curr.DebugString() << ", newestimation=" << m_est.DebugString() << ", numIter=" << m_numIters;
}

} // namespace salus
