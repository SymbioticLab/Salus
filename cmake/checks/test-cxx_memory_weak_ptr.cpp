#include <memory>

int main()
{
  bool isValid(false);

  std::weak_ptr<int> bar;

  {
    std::shared_ptr<int> foo(new int(42));
    bar = foo;
    if (auto spHandle = bar.lock()) {
      isValid = (*spHandle) == 42 ? true : false;
    }
  }

  return bar.expired() && isValid  ? 0 : 1;
}
