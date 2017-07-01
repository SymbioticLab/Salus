#include <memory>

int main()
{
  std::unique_ptr<int> foo = std::make_unique<int>(42);

  return ((*foo) == 42) ? 0 : 1;
}
