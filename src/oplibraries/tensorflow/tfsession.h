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

#include "utils/macros.h"
#include "utils/pointerutils.h"
#include "execution/executionengine.h"
#include "oplibraries/tensorflow/tfutils.h"
#include <memory>

namespace symbiotic::salus::oplib::tensorflow {
class TFInstance;

/**
 * @brief One session of job
 */
class TFSession : public std::enable_shared_from_this<TFSession>
{
public:
    TFSession(TFInstance &inst, ExecutionContext &&ctx, const tf::ConfigProto &config, tf::GraphDef *gdef);
    ~TFSession();

    std::string handle() const;

    /**
     * @brief Safely close the session.
     *
     * The resource is deleted later when execution engine actually removes internal structure.
     */
    void safeClose();

#define DECLARE_HANDLER(name) \
    void handle ## name (ZmqServer::Sender sender, const tf:: name ## Request &req, tf:: name ## Response &resp, StatusCallback &&cb)

    DECLARE_HANDLER(ExtendSession);
    DECLARE_HANDLER(PartialRunSetup);
    DECLARE_HANDLER(RunStep);

#undef DECLARE_HANDLER

private:
    class TFSessionPrivate;
    utils::PImpl<TFSessionPrivate> d;

    SALUS_DISALLOW_COPY_AND_ASSIGN(TFSession);
};

} // namespace symbiotic::salus::oplib::tensorflow

#endif // SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFSESSION_H
