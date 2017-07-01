#include <iostream>
#include <string>
#include <thread>

void myCallable(const std::string& id) {
  std::cout << "hello from id:" << id << std::endl;
}

int main(int argc, char *argv[]) {

  std::thread firstThread(myCallable,"first");
  std::thread secondThread(myCallable,"second");

  firstThread.join();
  secondThread.join();

  return 0;
}
