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

#ifndef SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFUTILS_H
#define SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFUTILS_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "utils/macros.h"

namespace symbiotic::salus::oplib::tensorflow {

using Status = ::tensorflow::Status;

class TFException : std::exception
{
    Status m_status;

public:
    explicit TFException(const Status &code);
    ~TFException();

    const char *what() const override;
}

} // namespace symbiotic::salus::oplib::tensorflow

#define SALUS_THROW_IF_ERROR(...)                                                                            \
    do {                                                                                                     \
        const ::tensorflow::Status _status = (__VA_ARGS__);                                                  \
        if (SALUS_PREDICT_FALSE(!_status.ok()))                                                              \
            throw symbiotic::salus::oplib::tensorflow::TFException(_status);                                 \
    } while (0)

#endif // SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFUTILS_H
