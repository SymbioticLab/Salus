/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Peifeng Yu <peifeng@umich.edu>
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

#ifndef SALUS_OPLIB_TENSORFLOW_TFEXCEPTION_H
#define SALUS_OPLIB_TENSORFLOW_TFEXCEPTION_H

#include "tensorflow_headers.h"
#include "tfutils.h"
#include "utils/macros.h"

namespace salus::oplib::tensorflow {

/**
 * @brief an exception encloses a ::tensorflow::Status
 */
class TFException : std::exception
{
    Status m_status;

public:
    explicit TFException(const Status &code);
    ~TFException() override;

    const Status &code() const
    {
        return m_status;
    }

    const char *what() const noexcept override;
};

} // namespace salus::oplib::tensorflow

#define SALUS_THROW_IF_ERROR(...)                                                                            \
    do {                                                                                                     \
        const auto _status = (__VA_ARGS__);                                                                  \
        if (SALUS_PREDICT_FALSE(!_status.ok()))                                                              \
            throw ::salus::oplib::tensorflow::TFException(_status);                                 \
    } while (0)

#endif // SALUS_OPLIB_TENSORFLOW_TFEXCEPTION_H
