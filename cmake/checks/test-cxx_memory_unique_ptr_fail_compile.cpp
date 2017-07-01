#include <memory>
#include <vector>

int main(int argc, const char *argv[])
{
  std::unique_ptr<int> foo(new int(42));
  std::vector<std::unique_ptr<int> > vec;
  vec.push_back(foo);
  return 1;
}
