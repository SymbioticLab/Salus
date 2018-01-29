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

#include <memory>

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

    auto pimpl() { return d.get(); }
    const auto pimpl() const { return d.get(); }

private:
    const std::unique_ptr<TFSessionPrivate> d;
};

} // namespace symbiotic::salus::oplib::tensorflow

#endif // SYMBIOTIC_SALUS_OPLIB_TENSORFLOW_TFSESSION_H
