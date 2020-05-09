#include "Utilities.cuh"

#include <stdio.h>

namespace StyLitCUDA {

void assertCUDA(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr, "epic CUDA fail: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

} /* namespace StyLitCUDA */
