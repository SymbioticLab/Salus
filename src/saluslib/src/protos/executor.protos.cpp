//
// Created by peifeng on 2/22/18.
//

#include "executor.protos.h"
#include "utils/stringutils.h"

namespace executor {

std::ostream &operator<<(std::ostream &os, const OpKernelDef &c)
{
    return os << "OpKernelDef(" << c.id() << ", " << executor::OpLibraryType_Name(c.oplibrary()) << ")";
}

std::ostream &operator<<(std::ostream &os, const EvenlopDef &c)
{
    return os << "EvenlopDef(type='" << c.type() << "', seq=" << c.seq() << ", sess=" << c.sessionid()
              << ", recvId='"
              << sstl::bytesToHexString(reinterpret_cast<const uint8_t *>(c.recvidentity().data()),
                                        c.recvidentity().size())
              << "')";
}

} // namespace executor
