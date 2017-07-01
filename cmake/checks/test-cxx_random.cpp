#include <random>

int main() {
  // Only test major requirements?
  std::mt19937 generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  double randNumber = distribution(generator);
  return 0;
}
