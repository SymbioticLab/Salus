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

#ifndef SALUS_SSTL_POINTERUTILS_H
#define SALUS_SSTL_POINTERUTILS_H

#include "platform/logging.h"
#include "utils/macros.h"
#include "utils/type_traits.h"

#include <deque>
#include <functional>
#include <memory>
#include <type_traits>

namespace sstl {

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
auto wrap_unique(T *ptr)
{
    static_assert(!std::is_array<T>::value, "array types are unsupported");
    static_assert(std::is_object<T>::value, "non-object types are unsupported");
    return std::unique_ptr<T>(ptr);
}

template<typename T>
struct ScopedUnref
{
    explicit ScopedUnref(T *o = nullptr)
        : obj(o)
    {
    }
    ~ScopedUnref()
    {
        if (obj)
            obj->Unref();
    }

    ScopedUnref(ScopedUnref &&other)
    {
        obj = other.obj;
        other.obj = nullptr;
    }

    ScopedUnref &operator=(ScopedUnref &&other)
    {
        obj = other.obj;
        other.obj = nullptr;
        return *this;
    }

    auto get() const
    {
        auto a = std::make_unique<int>(1);
        return obj;
    }

    operator bool() const
    {
        return obj != nullptr;
    }

private:
    T *obj;

    SALUS_DISALLOW_COPY_AND_ASSIGN(ScopedUnref);
};

template<typename T, typename... Args>
auto make_scoped_unref(Args &&... args)
{
    return ScopedUnref<T>(new T(std::forward<Args>(args)...));
}

template<typename T>
auto wrap_unref(T *ptr)
{
    static_assert(!std::is_array<T>::value, "array types are unsupported");
    static_assert(std::is_object<T>::value, "non-object types are unsupported");
    return ScopedUnref<T>(ptr);
}

class ScopeGuards
{
public:
    using CleanupFunction = std::function<void()>;

    ScopeGuards() = default;

    explicit ScopeGuards(CleanupFunction &&func)
    {
        *this += std::forward<CleanupFunction>(func);
    }

    template<typename Callable>
    ScopeGuards &operator+=(Callable &&undo_func)
    {
        funcs.emplace_front(std::forward<Callable>(undo_func));
        return *this;
    }

    void dismiss() noexcept
    {
        funcs.clear();
    }

    ~ScopeGuards()
    {
        for (auto &f : funcs)
            f(); // must not throw
    }

private:
    std::deque<CleanupFunction> funcs;

    ScopeGuards(const ScopeGuards &) = delete;
    void operator=(const ScopeGuards &) = delete;
};

/**
 * @brief A wrapper class for PIMPL pattern to make sure const correctness.
 */
template<typename TPriv>
class PImpl
{
public:
    static_assert(std::is_object_v<TPriv>, "TPriv must not be a function or a reference");

    using DType = std::decay_t<TPriv>;

    template<typename... Args,
             typename SFINAE = std::enable_if_t<
                 !(sizeof...(Args) == 1 && std::is_same_v<std::decay_t<arg_first_t<Args...>>, PImpl>)>>
    PImpl(Args &&... args)
        : m_priv(std::forward<Args>(args)...)
    {
    }

    /**
     * @brief Allow move
     */
    PImpl(PImpl &&) = default;
    PImpl &operator=(PImpl &&) = default;

    DType *operator->()
    {
        return m_priv.get();
    }
    const DType *operator->() const
    {
        return m_priv.get();
    }

private:
    std::unique_ptr<DType> m_priv;

    SALUS_DISALLOW_COPY_AND_ASSIGN(PImpl);
};

// Following is copied from Microsoft GSL library

///////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Microsoft Corporation. All rights reserved.
//
// This code is licensed under the MIT License (MIT).
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief owner<T> is designed as a bridge for code taht must deal directly with owning pointers for some
 * reason
 *
 * - Disalow construction from any type other than pointer type
 *
 * @tparam T must be a pointer type
 */
template<class T, class = std::enable_if_t<std::is_pointer<T>::value>>
using owner = T;

/**
 * @brief Restricts a pointer or smart pointer to only hold non-null values.
 *
 * @tparam T must be a pointer type
 *
 * Has zero size overhead over T.
 *
 * If T is a pointer (i.e. T == U*) then
 * - allow construction from U*
 * - disallow construction from nullptr_t
 * disallow default_construction
 * - ensure construction from null U* fails
 * - allow implicit conversion to U*
 */
template<class T>
class not_null
{
public:
    static_assert(std::is_assignable<T &, std::nullptr_t>::value, "T cannot be assigned nullptr.");

    template<typename U, typename = std::enable_if_t<std::is_convertible<U, T>::value>>
    constexpr not_null(U &&u)
        : ptr_(std::forward<U>(u))
    {
        DCHECK(ptr_ != nullptr);
    }

    template<typename U, typename = std::enable_if_t<std::is_convertible<U, T>::value>>
    constexpr not_null(const not_null<U> &other)
        : not_null(other.get())
    {
    }

    not_null(const not_null &other) = default;
    not_null &operator=(const not_null &other) = default;

    constexpr T get() const
    {
        DCHECK(ptr_ != nullptr);
        return ptr_;
    }

    constexpr operator T() const
    {
        return get();
    }
    constexpr T operator->() const
    {
        return get();
    }
    constexpr decltype(auto) operator*() const
    {
        return *get();
    }

    // prevents compilation when someone attempts to assign a null pointer constant
    not_null(std::nullptr_t) = delete;
    not_null &operator=(std::nullptr_t) = delete;

    // unwanted operators...pointers only point to single objects!
    not_null &operator++() = delete;
    not_null &operator--() = delete;
    not_null operator++(int) = delete;
    not_null operator--(int) = delete;
    not_null &operator+=(std::ptrdiff_t) = delete;
    not_null &operator-=(std::ptrdiff_t) = delete;
    void operator[](std::ptrdiff_t) const = delete;

private:
    T ptr_;
};

template<class T>
std::ostream &operator<<(std::ostream &os, const not_null<T> &val)
{
    os << val.get();
    return os;
}

template<class T, class U>
auto operator==(const not_null<T> &lhs, const not_null<U> &rhs) -> decltype(lhs.get() == rhs.get())
{
    return lhs.get() == rhs.get();
}

template<class T, class U>
auto operator!=(const not_null<T> &lhs, const not_null<U> &rhs) -> decltype(lhs.get() != rhs.get())
{
    return lhs.get() != rhs.get();
}

template<class T, class U>
auto operator<(const not_null<T> &lhs, const not_null<U> &rhs) -> decltype(lhs.get() < rhs.get())
{
    return lhs.get() < rhs.get();
}

template<class T, class U>
auto operator<=(const not_null<T> &lhs, const not_null<U> &rhs) -> decltype(lhs.get() <= rhs.get())
{
    return lhs.get() <= rhs.get();
}

template<class T, class U>
auto operator>(const not_null<T> &lhs, const not_null<U> &rhs) -> decltype(lhs.get() > rhs.get())
{
    return lhs.get() > rhs.get();
}

template<class T, class U>
auto operator>=(const not_null<T> &lhs, const not_null<U> &rhs) -> decltype(lhs.get() >= rhs.get())
{
    return lhs.get() >= rhs.get();
}

// more unwanted operators
template<class T, class U>
std::ptrdiff_t operator-(const not_null<T> &, const not_null<U> &) = delete;
template<class T>
not_null<T> operator-(const not_null<T> &, std::ptrdiff_t) = delete;
template<class T>
not_null<T> operator+(const not_null<T> &, std::ptrdiff_t) = delete;
template<class T>
not_null<T> operator+(std::ptrdiff_t, const not_null<T> &) = delete;

} // namespace sstl

namespace std {
template<class T>
struct hash<::sstl::not_null<T>>
{
    std::size_t operator()(const ::sstl::not_null<T> &value) const
    {
        return hash<T>{}(value);
    }
};

} // namespace std

#endif // SALUS_SSTL_POINTERUTILS_H
