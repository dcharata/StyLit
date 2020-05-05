#include <stdio.h>

#include "ConnectDownscalerCUDA.h"

__global__ void cudaHelloKernel(){
  printf("Hello from mykernel\n");
}

int glue() {
  cudaHelloKernel<<<1,1>>>();
  cudaDeviceSynchronize();
  return 0;
}
