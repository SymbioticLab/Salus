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

#ifndef SALUS_OPLIB_TENSORFLOW_TFEXCEPTION_H
#define SALUS_OPLIB_TENSORFLOW_TFEXCEPTION_H

#include "oplibraries/tensorflow/tensorflow_headers.h"
#include "oplibraries/tensorflow/tfutils.h"
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
