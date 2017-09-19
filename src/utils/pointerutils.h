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
 *
 */

#ifndef POINTERUTILS_H
#define POINTERUTILS_H

#include <memory>

namespace utils {

template<typename Derived, typename Base>
std::unique_ptr<Derived> static_unique_ptr_cast(std::unique_ptr<Base> &&p)
{
    auto d = static_cast<Derived *>(p.release());
    return std::unique_ptr<Derived>(d);
}

template<typename Derived, typename Base>
std::unique_ptr<Derived> dynamic_unique_ptr_cast(std::unique_ptr<Base> &&p)
{
    if (Derived *result = dynamic_cast<Derived *>(p.get())) {
        p.release();
        return std::unique_ptr<Derived>(result);
    }
    return std::unique_ptr<Derived>(nullptr);
}

template<typename T>
struct ScopedUnref
{
    explicit ScopedUnref(T *o = nullptr) : obj(o) {}
    ~ScopedUnref() {
        if (obj) obj->Unref();
    }

    ScopedUnref(ScopedUnref &&other) {
        obj = other.obj;
        other.obj = nullptr;
    }

    ScopedUnref &operator=(ScopedUnref &&other) {
        auto tmp(std::move(other));
        using std::swap;
        swap(*this, tmp);
        return *this;
    }

    auto get() const {
        auto a = std::make_unique<int>(1);
        return obj;
    }

    operator bool() const {
        return obj != nullptr;
    }

private:
    T *obj;

    ScopedUnref(const ScopedUnref&) = delete;
    ScopedUnref<T> & operator=(const ScopedUnref&) = delete;
};

template<typename T, typename... Args>
auto make_scoped_unref(Args && ...args)
{
    return ScopedUnref<T>(new T(std::forward<Args>(args)...));
}

} // namespace utils

#endif // POINTERUTILS_H
