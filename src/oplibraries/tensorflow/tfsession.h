/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Aetf <aetf@unlimitedcodeworks.xyz>
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

#ifndef SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFSESSION_H
#define SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFSESSION_H

#include "utils/pointerutils.h"
#include <memory>

#define CallWithMasterMethodName(m)                                                                          \
    m(CreateSession) m(ExtendSession) m(PartialRunSetup) m(CloseSession) m(ListDevices) m(Reset)

#define CallWithAllMethodName(m)                                                                             \
    CallWithMasterMethodName(m) m(RunStep)

namespace tensorflow {
#define FWD_DECLARE(name)                                                                                    \
    class name##Request;                                                                                     \
    class name##Response;

CallWithAllMethodName(FWD_DECLARE)

#undef FWD_DECLARE
} // namespace tensorflow

namespace symbiotic::salus::oplib::tensorflow {
class TFInstance;

/**
 * @brief One session of job
 */
class TFSession
{
public:
    explicit TFSession(TFInstance &inst);
    ~TFSession();


#define DECLARE_HANDLER(name) \
    void Handle ## name (const name ## Request *req, std::function<void(name ## Response*, Status)> cb);

    CallWithMasterMethodName(DECLARE_HANDLER)
    DECLARE_HANDLER(RunStep)

#undef DECLARE_HANDLER

private:
    class TFSessionPrivate;
    utils::PImpl<TFSessionPrivate> d;
};

} // namespace symbiotic::salus::oplib::tensorflow

#endif // SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFSESSION_H
