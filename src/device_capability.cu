#include <iostream>

int main(void) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "CC: " << deviceProp.major << "." << deviceProp.minor << "\n";

  return 0;
}
