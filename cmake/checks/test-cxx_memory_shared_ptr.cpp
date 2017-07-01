#include <memory>

int main()
{
  std::shared_ptr<int> foo(new int(42));
  std::shared_ptr<int> bar = foo;
  return (foo == bar) && ((*foo) == (*bar)) ? 0 : 1;
}
