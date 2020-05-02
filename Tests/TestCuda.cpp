#include "TestCuda.h"

#include <iostream>
#include <cuda_runtime.h>

extern "C"
int vectorAddmain(void);

bool TestCuda::run() {
  int status = vectorAddmain();
  if (status != 0)
    std::cout << "CUDA Error." << std::endl;
  else
    std::cout<<"CUDA WORKING!" << std::endl;

  return true;
}
