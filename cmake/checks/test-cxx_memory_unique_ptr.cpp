#include <memory>

int main()
{
  std::unique_ptr<int> foo(new int(42));
  std::unique_ptr<int> bar(new int(42));

  return ((foo != bar) && ((*foo) == (*bar))) ? 0 : 1;
}
