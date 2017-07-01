#include <functional>

enum class Foo
{
    Bar,
    Zee
};

int main()
{
    std::hash<Foo>{}(Foo::Bar);
    return 0;
}
