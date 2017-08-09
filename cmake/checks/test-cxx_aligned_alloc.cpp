#include <cstdlib>

int main()
{
    auto ptr = std::aligned_alloc(128, 10);
    std::free(ptr);
    return 0;
}
