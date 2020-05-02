#include "TestCuda.h"

#include <iostream>
#include <cuda_runtime.h>

extern "C"
    cudaError_t cuda_main();

bool TestCuda::run() {
  cudaError_t cuerr = cuda_main();
  if (cuerr != cudaSuccess)
    std::cout << "CUDA Error: " << cudaGetErrorString( cuerr ) << std::endl;
  else
    std::cout<<"CUDA WORKING!" << std::endl;

  return true;
}
